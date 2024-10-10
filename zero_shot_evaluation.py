import os
import time
import sys

import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from datasets import load_dataset, load_from_disk, Audio
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group, gather
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.gamma import Gamma
from sklearn.metrics import accuracy_score
from audiotools import AudioSignal

from model.rawnet3 import MainModel
from model.m5 import *
from model.resnet_2d import ResNet50_2D
from model.whole_system import WholeSystem
from attack.pgd import PGD
from attack.fakebob import FakeBob
import dac

# seed
torch.manual_seed(42)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="nccl", type=str)
    # dataset configuration
    parser.add_argument("--dataset", default="vctk", choices=["vctk", "speech_command", "esc50"])
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--pin_memory", default=True, type=bool)

    # evaluating configuration
    parser.add_argument("--defense_model_task", default="rvq", choices=["rvq", "rvq_denoise"])
    parser.add_argument("--noise_level", default=50, type=int)
    parser.add_argument("--defense_model_path", default=None, type=str)
    parser.add_argument("--classifier_path", default=None, type=str)

    # attack configuration
    parser.add_argument("--budget", default=50, type=int)
    parser.add_argument("--attack_type", default="pgd", choices=["pgd", "fakebob"], type=str)
    parser.add_argument("--epsilon", default=8/255, type=float)
    parser.add_argument("--norm", default="inf", choices=["inf", "l1", "l2"], type=str)
    parser.add_argument("--adv_inf_results_path", type=str)
    parser.add_argument("--benign_inf_results_path", type=str)
    parser.add_argument("--find_unused_parameters", default=False, type=bool)
    parser.add_argument("--task_label_path", type=str)
    return parser


def waveform_collate_fn(batch):
    padding_result = {}
    lengths = [data['audio']['array'].shape[-1] for data in batch]
    max_len = max(lengths)
    pad_counts = [max_len - length for length in lengths]
    padding_result["input_values"] = torch.tensor(
        [torch.cat([
            torch.from_numpy(data["audio"]["array"]).float(), 
            torch.zeros(max_len - length)
            ], 
            dim=1
            ) 
         for data, length in zip(batch, lengths)
         ]
         )
    padding_result["labels"] = torch.tensor([data["label"] for data in batch])
    return padding_result

def ddp_setup():
   """
   Args:
       rank: Unique identifier of each process
       world_size: Total number of processes
   """
   os.environ["MASTER_ADDR"] = "localhost"
   os.environ["MASTER_PORT"] = "12355"
   init_process_group(backend="nccl")

def main(args):
    # device
    rank = dist.get_rank()
    device_id = rank % torch.cuda.device_count()

    # print nothing if rank != 0
    if rank != 0:
        f = open(os.devnull, 'w')
        sys.stdout = f
    # device_id = "0"
    # if torch.cuda.is_available():
    #     device = f"cuda:{device_id}"
    # else:
    #     device = "cpu"
    # print(f"using {device}")

    # dataset, need to implement for specific dataset

    if args.dataset == "speech_commands":
        whole_set = load_dataset("google/speech_commands", "v0.02")
        labels = whole_set['train'].unique("label")
        classes = len(labels)
        classifier = CNN(
            channels = [[48], [48]*3, [96]*4, [192]*6, [384]*3],
            conv_kernels = [80, 3, 3, 3, 3],
            conv_strides = [4, 1, 1, 1, 1],
            conv_padding = [38, 1, 1, 1, 1],
            pool_padding = [0, 0, 0, 2],
            num_classes=len(classes)
        )
    elif args.dataset == "esc50":
        whole_set = load_dataset("ashraq/esc50")
        whole_set = whole_set.rename_column("category", "label")
        classes = 50
        labels = whole_set["train"].unique("label")

    
    evaluation_set = whole_set["test"]
    sampler = DistributedSampler(evaluation_set)
    eval_loader = DataLoader(evaluation_set, batch_size=args.batch_size, sampler=sampler,
                            num_workers=args.num_workers, pin_memory=True,
                            collate_fn=waveform_collate_fn)
        
    print(f"effective batch size is {args.batch_size}")

    # model
    model_path = dac.utils.download(model_type="44khz")
    defense_model = dac.DAC.load(model_path)
    

    # ddp_defense_model = DistributedDataParallel(defense_model, device_ids=[rank], output_device=rank, find_unused_parameters=args.find_unused_parameters)
    # ddp_classifier = DistributedDataParallel(classifier, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    print(f"Defense Model is: {defense_model}")
    print(f"Classifier is: {classifier}") 


    classifier_checkpoint = torch.load(args.classifier_path, map_location=f"cuda:{device_id}")
    state_dict = {}
    for key in classifier_checkpoint["model"]:
        new_key = key.replace("module.", "")
        state_dict[new_key] = classifier_checkpoint["model"][key]
    classifier.load_state_dict(state_dict)

    print(f"load defense model from {args.defense_model_path}")
    print(f"load classifier from {args.classifier_path}")

    wholesystem = WholeSystem(
        defense_model=defense_model, 
        classifier=classifier,
        original_sampling_rate=44000,
        new_sampling_rate=8000
        ).to(f"cuda:{device_id}")
    ddp_wholesystem = DistributedDataParallel(wholesystem, device_ids=[rank], output_device=rank, find_unused_parameters=args.find_unused_parameters)

    # adversarial attack
    if args.norm == "norm":
        norm = np.inf
    elif args.norm == "l1":
        norm = 1
    elif args.norm == "l2":
        norm = 2

    if args.attack_type == "pgd":  
        attack = PGD(
            model=ddp_wholesystem,
            eps=args.epsilon,
            steps=args.budget,
        )
    elif args.attack_type == "fakebob":
        attack = FakeBob(
            model=ddp_wholesystem,
            lr=1e-4,
            steps=200,
            epsilon=0.002,
            norm_type=norm
        )

    # evaluation loop
    print(f"start evaluation!")

    labels = np.array([])
    inference_result = np.array([])
    adv_inference_result = np.array([])

    ddp_wholesystem.eval()
    with tqdm(total=len(eval_loader)) as eval_pbar:
        for step, data in enumerate(eval_loader):
            # to device
            data['input_values'] = defense_model.preprocess(data['input_values']).to(f"cuda:{device_id}")
            data['labels'] = data['labels'].to(f"cuda:{device_id}")
                
            # generate adversarial examples
            adv_input_data = attack(
                data['input_values'], 
                data['labels']
            )
            # input to the model
            adv_logits = ddp_wholesystem(adv_input_data['input_values'])
            adv_classification_result = torch.argmax(adv_logits, dim=-1)
                
            # benign examples
            logits = ddp_wholesystem(data['input_values'])
            classification_result = torch.argmax(logits, dim=-1)
            batch_labels = data['labels']

            # inter process comunication
            if rank != 0:
                gather(classification_result, dst=0)
                gather(adv_classification_result, dst=0)
                gather(batch_labels, dst=0)
                
            else:
                classification_results = [torch.empty_like(classification_result) for _ in range(dist.get_world_size())]
                adv_classification_results  = [torch.empty_like(adv_classification_result) for _ in range(dist.get_world_size())]
                batch_labels_list = [torch.empty_like(batch_labels) for _ in range(dist.get_world_size())]
            
                gather(classification_result, gather_list=classification_results)
                gather(adv_classification_result, gather_list=adv_classification_results)
                gather(batch_labels, gather_list=batch_labels_list)
                for result_and_label in zip(classification_results, adv_classification_results, batch_labels_list):
                    inference_result = np.concatenate([inference_result, result_and_label[0].cpu().numpy()])
                    adv_inference_result = np.concatenate([adv_inference_result, result_and_label[1].cpu().numpy()])
                    labels = np.concatenate([labels, result_and_label[2].cpu().numpy()])


            # progress bar
            current_robust_acc = accuracy_score(adv_inference_result, labels)
            current_benign_acc = accuracy_score(inference_result, labels)
            eval_pbar.set_description(f"Step: {step+1}/{len(eval_loader)}")
            eval_pbar.set_postfix(r_acc="{:.4f}".format(current_robust_acc), b_acc="{:.4f}".format(current_benign_acc))
            eval_pbar.update(1)

    adv_acc = accuracy_score(adv_inference_result, labels)
    acc = accuracy_score(inference_result, labels)

    print(f"the robust accuracy is: {adv_acc}")
    print(f"the standard accuracy is: {acc} ")

    if rank == 0:
        with open(args.adv_inf_results_path, "wb") as f:
            np.save(f, adv_inference_result)

        with open(args.benign_inf_results_path, "wb") as f:
            np.save(f, inference_result)
            
        with open(args.task_label_path, "wb") as f:
            np.save(f, labels)
        
if __name__ == "__main__":
    parser = get_args()
    args = parser.parse_args()
    ddp_setup()
    main(args)
