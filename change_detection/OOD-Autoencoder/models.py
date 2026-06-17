import torch
import torch.nn as nn
import torchkbnufft as tkbn

class FrequencyAutoencoder(nn.Module):
    def __init__(self, sequence_length, latent_dim):
        """
        Autoencoder that works strictly in the frequency domain.
        It uses the Fast Fourier Transform to extract the amplitude spectrum
        and reconstructs it.
        """
        super(FrequencyAutoencoder, self).__init__()
        
        # Length of the output from the NUFFT slice [:, sequence_length // 2:]
        self.freq_length = sequence_length - (sequence_length // 2)
        
        # NUFFT Object (Type 1: Non-uniform to Uniform)
        self.nufft = tkbn.KbNufftAdjoint(im_size=(sequence_length,))
        
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

    def forward(self, points, values):
        """
        Forward pass.
        Args:
            points: Input scaled time coordinates of shape (batch_size, sequence_length)
            values: Input valid data of shape (batch_size, sequence_length)
        Returns:
            freq_amplitudes: True NUFFT amplitude spectrum
            reconstructed_amplitudes: Model's reconstructed amplitude spectrum
        """
        # 1. Compute the Type 1 Non-Uniform Fast Fourier Transform (NUFFT)
        sequence_length = points.shape[-1]
        
        # torchkbnufft expects points shape (batch_size, 1, sequence_length)
        # or (1, 1, sequence_length) if identical across batch
        pts_1d = points[0].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, sequence_length)
        
        # values shape needs to be (batch_size, 1, sequence_length) and complex64
        val_1d = values.unsqueeze(1).to(torch.complex64)
        
        # Compute NUFFT. Output shape: (batch_size, 1, sequence_length)
        freq_complex = self.nufft(val_1d, pts_1d).squeeze(1)
        
        # Slicing the positive frequencies (and DC) to match rfft shape
        freq_complex = freq_complex[:, sequence_length // 2:]
        
        # 2. Extract Amplitude Spectrum
        freq_amplitudes = torch.abs(freq_complex)
        
        # 3. Autoencode
        latent = self.encoder(freq_amplitudes)
        reconstructed_amplitudes = self.decoder(latent)
        
        return freq_amplitudes, reconstructed_amplitudes
