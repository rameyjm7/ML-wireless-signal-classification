import torch
import torch.nn as nn
import torch.fft
import numpy as np
from scipy.signal import convolve


class AdaptiveNoiseReduction(nn.Module):
    def __init__(self, threshold=0.1):
        super(AdaptiveNoiseReduction, self).__init__()
        self.threshold = threshold  # Factor to scale the noise threshold

    def forward(self, iq_data):
        # Step 1: Calculate the noise level estimate from the IQ signal
        noise_estimate = torch.std(iq_data, dim=-1, keepdim=True)

        # Step 2: Perform soft thresholding in the frequency domain
        iq_fft = torch.fft.fft(iq_data, dim=-1)
        magnitude = torch.abs(iq_fft)
        
        # Adaptive threshold based on noise estimate
        threshold = self.threshold * noise_estimate
        
        # Apply soft thresholding: reduce noise while preserving signal structure
        iq_fft_denoised = torch.where(magnitude > threshold, iq_fft, torch.zeros_like(iq_fft))
        
        # Step 3: Inverse FFT to return to time domain
        iq_denoised = torch.fft.ifft(iq_fft_denoised, dim=-1).real
        return iq_denoised
    
    def soft_threshold(self, x):
        """Applies soft thresholding to the input data."""
        return np.sign(x) * np.maximum(np.abs(x) - self.threshold, 0)

    def apply(self, signal):
        """Applies ANR with 1D convolutional feature extraction."""
        # Ensure the signal is 1D
        signal = np.asarray(signal).flatten()

        # Use 1D kernels for convolution
        kernel_3 = np.ones(3) / 3  # 1D kernel with size 3
        kernel_5 = np.ones(5) / 5  # 1D kernel with size 5

        # Perform convolution with 1D kernels
        feature_3 = convolve(signal, kernel_3, mode='same')
        feature_5 = convolve(signal, kernel_5, mode='same')

        # Combine features and apply soft threshold
        combined_features = feature_3 + feature_5
        denoised_output = self.soft_threshold(combined_features)

        return denoised_output
    