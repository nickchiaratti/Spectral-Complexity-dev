import torch
import torch.nn as nn

class FrequencyAutoencoder(nn.Module):
    def __init__(self, sequence_length, latent_dim):
        """
        Autoencoder that works strictly in the frequency domain.
        It uses the Fast Fourier Transform to extract the amplitude spectrum
        and reconstructs it.
        """
        super(FrequencyAutoencoder, self).__init__()
        
        # Length of the output from real FFT (rfft)
        self.freq_length = sequence_length // 2 + 1
        
        # Encoder: Fully connected layers to reduce dimensionality
        self.encoder = nn.Sequential(
            nn.Linear(self.freq_length, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )
        
        # Decoder: Reconstructs the amplitude spectrum
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, self.freq_length),
            # Amplitude spectrum is strictly non-negative, so we use ReLU at the end
            nn.ReLU()
        )

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: Input time series tensor of shape (batch_size, sequence_length)
        Returns:
            freq_amplitudes: True FFT amplitude spectrum
            reconstructed_amplitudes: Model's reconstructed amplitude spectrum
        """
        # 1. Compute the Real Fast Fourier Transform
        freq_complex = torch.fft.rfft(x, dim=1)
        
        # 2. Extract Amplitude Spectrum
        freq_amplitudes = torch.abs(freq_complex)
        
        # 3. Autoencode
        latent = self.encoder(freq_amplitudes)
        reconstructed_amplitudes = self.decoder(latent)
        
        return freq_amplitudes, reconstructed_amplitudes
