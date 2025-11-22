"""Vocos vocoder wrapper for Matcha-TTS."""

import torch
import torch.nn as nn
from vocos import Vocos

class VocosWrapper(nn.Module):
    """
    Wrapper for Vocos vocoder to make it compatible with Matcha-TTS interface.
    Vocos is a mel-spectrogram vocoder that can be used as a drop-in replacement for HiFiGAN.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, mel):
        """
        Forward pass through the vocoder.
        
        Args:
            mel: Mel-spectrogram tensor of shape (batch, n_mels, time)
        
        Returns:
            audio: Audio waveform tensor of shape (batch, 1, samples)
        """

        with torch.no_grad():
            audio = self.model.decode(mel)
        
        return audio


def load_vocoder(model_id="BSC-LT/vocos-mel-22khz", device="cuda"):
    """
    Load a pretrained Vocos model from HuggingFace  
    
    Args:
        model_id: a hugging face model ID, e.g. "BSC-LT/vocos-mel-22khz" 
                  (see https://huggingface.co/BSC-LT/vocos-mel-22khz)
        device: Device to load the model on (cpu/cuda)
    
    Returns:
        VocosWrapper instance
    """
    
    vocos = Vocos.from_pretrained(model_id)
    vocos = vocos.eval().to(device)
    
    return VocosWrapper(vocos)
