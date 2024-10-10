import torch
import torch.nn as nn
from torchaudio.transforms import Resample

class WholeSystem(nn.Module):
    def __init__(self, defense_model, classifier, original_sampling_rate=44000, new_sampling_rate=8000) -> None:
        super(WholeSystem, self).__init__()
        self.defense_model = defense_model
        self.classifier = classifier
        self.resample = Resample(
            orig_freq=original_sampling_rate,
            new_freq=new_sampling_rate
        )

    def forward(self, input_data):
        reconstructed_input = self.defense_model(input_data)
        reconstructed_input = self.resample(reconstructed_input)
        classification_result = self.classifier(reconstructed_input)
        return classification_result