import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from typing import List


# Model configuration
CHUNK_SIZE = 1300  # 10s * 130Hz (Model window size)


class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm and LeakyReLU."""
    
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_c),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm1d(out_c),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class AttentionGate(nn.Module):
    """Attention Gate for U-Net skip connections."""
    
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv1d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv1d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv1d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttentionUNet(nn.Module):
    """Attention U-Net architecture for ECG signal denoising."""
    
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(1, 32)
        self.pool1 = nn.MaxPool1d(2)
        self.enc2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool1d(2)
        self.enc3 = ConvBlock(64, 128)
        self.pool3 = nn.MaxPool1d(2)
        self.enc4 = ConvBlock(128, 256)
        self.pool4 = nn.MaxPool1d(2)
        
        # Center
        self.center = ConvBlock(256, 512)
        
        # Decoder with Attention Gates
        self.up4 = nn.ConvTranspose1d(512, 256, 2, stride=2)
        self.att4 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.dec4 = ConvBlock(512, 256)
        
        self.up3 = nn.ConvTranspose1d(256, 128, 2, stride=2)
        self.att3 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.dec3 = ConvBlock(256, 128)
        
        self.up2 = nn.ConvTranspose1d(128, 64, 2, stride=2)
        self.att2 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.dec2 = ConvBlock(128, 64)
        
        self.up1 = nn.ConvTranspose1d(64, 32, 2, stride=2)
        self.att1 = AttentionGate(F_g=32, F_l=32, F_int=16)
        self.dec1 = ConvBlock(64, 32)
        
        # Final output
        self.final = nn.Conv1d(32, 1, 1)

    def forward(self, x):
        orig_len = x.shape[-1]
        pad_len = (16 - orig_len % 16) % 16
        if pad_len > 0:
            x = F.pad(x, (0, pad_len))
        
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)
        
        # Center
        c = self.center(p4)
        
        # Decoder with attention
        d4 = self.up4(c)
        x4 = self.att4(g=d4, x=e4)
        d4 = torch.cat((x4, d4), dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        x3 = self.att3(g=d3, x=e3)
        d3 = torch.cat((x3, d3), dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        x2 = self.att2(g=d2, x=e2)
        d2 = torch.cat((x2, d2), dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        x1 = self.att1(g=d1, x=e1)
        d1 = torch.cat((x1, d1), dim=1)
        d1 = self.dec1(d1)
        
        out = self.final(d1)
        return out[..., :orig_len]


class ECGModel:
    """Singleton wrapper for AttentionUNet model inference. Handles model loading and ECG signal denoising."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ECGModel, cls).__new__(cls)
            cls._instance._model = None
            cls._instance._device = None
        return cls._instance

    def load(self, model_path: str) -> None:
        """Load the AttentionUNet model weights."""
        if self._model is None:
            try:
                # Determine device (CPU or CUDA)
                self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                # Initialize model
                self._model = AttentionUNet().to(self._device)

                # Load weights
                if os.path.exists(model_path):
                    self._model.load_state_dict(
                        torch.load(model_path, map_location=self._device)
                    )
                    self._model.eval()
                    print(f"✅ AttentionUNet model loaded from: {model_path}")
                else:
                    raise FileNotFoundError(f"Model file not found: {model_path}")

            except Exception as e:
                raise RuntimeError(f"Failed to load AttentionUNet model: {str(e)}")

    def denoise(self, ecg_signal: np.ndarray) -> np.ndarray:
        """Denoise ECG signal using AttentionUNet model with Z-score normalization and chunk processing."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # 1. Z-score normalization of the entire signal
        mean = np.mean(ecg_signal)
        std = np.std(ecg_signal)
        if std == 0:
            std = 1
        normalized_signal = (ecg_signal - mean) / std

        cleaned_chunks = []

        # 2. Process in chunks (chunking)
        num_samples = len(normalized_signal)
        self._model.eval()

        with torch.no_grad():
            for i in range(0, num_samples, CHUNK_SIZE):
                chunk = normalized_signal[i: i + CHUNK_SIZE]

                # Padding if last chunk is less than CHUNK_SIZE samples
                original_len = len(chunk)
                if original_len < CHUNK_SIZE:
                    pad_width = CHUNK_SIZE - original_len
                    chunk = np.pad(chunk, (0, pad_width), 'constant')

                # Convert to Tensor [Batch=1, Channel=1, Length=CHUNK_SIZE]
                input_tensor = torch.from_numpy(chunk).float().view(1, 1, CHUNK_SIZE).to(self._device)

                # Inference
                output_tensor = self._model(input_tensor)

                # Get result back to numpy
                cleaned_chunk = output_tensor.cpu().numpy().flatten()

                # Remove padding (if last chunk)
                cleaned_chunk = cleaned_chunk[:original_len]
                cleaned_chunks.append(cleaned_chunk)

        # 3. Concatenate results
        full_clean_signal = np.concatenate(cleaned_chunks)

        return full_clean_signal
