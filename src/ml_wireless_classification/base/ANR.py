import torch
import torch.nn as nn
import torch.fft

class AdaptiveNoiseReduction(nn.Module):
    def __init__(self, threshold_scale=0.1):
        super(AdaptiveNoiseReduction, self).__init__()
        self.threshold_scale = threshold_scale  # Factor to scale the noise threshold

    def forward(self, iq_data):
        # Step 1: Calculate the noise level estimate from the IQ signal
        noise_estimate = torch.std(iq_data, dim=-1, keepdim=True)

        # Step 2: Perform soft thresholding in the frequency domain
        iq_fft = torch.fft.fft(iq_data, dim=-1)
        magnitude = torch.abs(iq_fft)
        
        # Adaptive threshold based on noise estimate
        threshold = self.threshold_scale * noise_estimate
        
        # Apply soft thresholding: reduce noise while preserving signal structure
        iq_fft_denoised = torch.where(magnitude > threshold, iq_fft, torch.zeros_like(iq_fft))
        
        # Step 3: Inverse FFT to return to time domain
        iq_denoised = torch.fft.ifft(iq_fft_denoised, dim=-1).real
        return iq_denoised
