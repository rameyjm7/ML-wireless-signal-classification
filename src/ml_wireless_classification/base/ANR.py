import torch
import torch.nn as nn
import torch.fft
import numpy as np
from scipy.signal import convolve
from scipy.ndimage import gaussian_filter1d

class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block for adaptive weighting of channels."""
    def __init__(self, in_channels, reduction=4):
        super(SqueezeExcitation, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)

    def forward(self, x):
        # Global average pooling (average across the temporal dimension)
        scale = x.mean(dim=-1, keepdim=True)
        
        # Fully connected layers for adaptive scaling
        scale = torch.relu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))

        # Scale the input x by the learned weights
        return x * scale

class AdaptiveNoiseReduction(nn.Module):
    def __init__(self, num_channels=4, threshold=0.05, method="convolution"):
        super(AdaptiveNoiseReduction, self).__init__()
        self.num_channels = num_channels  # Set the number of ANR channels
        self.threshold = threshold  # Factor to scale the noise threshold
        self.method = method  # Specify the method: "convolution" or "gaussian"

    def forward(self, iq_data):
        # Step 1: Calculate the noise level estimate from the IQ signal
        noise_estimate = torch.std(iq_data, dim=-1, keepdim=True)

        # Step 2: Perform soft thresholding in the frequency domain
        iq_fft = torch.fft.fft(iq_data, dim=-1)
        magnitude = torch.abs(iq_fft)

        # Adaptive threshold based on noise estimate and channel configuration
        threshold = self.threshold * noise_estimate * (1 / self.num_channels)  # Scale by num_channels

        # Apply soft thresholding: reduce noise while preserving signal structure
        iq_fft_denoised = torch.where(magnitude > threshold, iq_fft, torch.zeros_like(iq_fft))

        # Step 3: Inverse FFT to return to time domain
        iq_denoised = torch.fft.ifft(iq_fft_denoised, dim=-1).real
        return iq_denoised


    def soft_threshold(self, x):
        """Applies soft thresholding to the input data."""
        return np.sign(x) * np.maximum(np.abs(x) - self.threshold, 0)

    def apply(self, signal, sigma=2):
        """Applies ANR with either convolution-based denoising or Gaussian denoising based on the selected method."""
        denoised_signal = signal.clone()

        # Apply the denoising multiple times if `num_channels` > 1
        for _ in range(self.num_channels):
            if self.method == "convolution":
                # Use 1D kernel convolution
                kernel = np.ones(3) / 3
                denoised_signal = torch.tensor(convolve(denoised_signal.numpy(), kernel, mode='same'))
            elif self.method == "gaussian":
                # Apply Gaussian smoothing with adaptive sigma
                denoised_signal = gaussian_filter1d(denoised_signal.numpy(), sigma=sigma)
                denoised_signal = torch.tensor(denoised_signal)
        
        return denoised_signal

