# Heartify - ECG Denoising Model API

> Flask API backend for ECG signal denoising using Attention U-Net deep learning architecture

## 📋 Overview

Heartify Denoising API provides a REST API for cleaning noisy ECG signals using an **Attention U-Net** model. The system removes noise from raw ECG signals captured from devices like Polar H10, producing clean signals suitable for further analysis.

**Key Features:**
- 🔐 API Key management with email verification
- 🧹 ECG signal denoising (1-lead, 130Hz signal processing)
- 🤖 Attention U-Net architecture for high-quality noise removal
- 🔒 Secure authentication for all denoising requests

## 🚀 API Endpoints

### API Key Management

**POST** `/api/v1/api-keys/generation` - Request new API key  
**POST** `/api/v1/api-keys/deactivation` - Deactivate existing API key  
**GET** `/api/v1/api-keys/verify?token=...` - Verify email

### ECG Denoising

**POST** `/api/v1/denoising`

Denoise ECG signal and return cleaned signal.

**Headers:**
```
x-api-key: your-api-key
```

**Request:**
```json
{
  "ecg_signal": [array of 1300 float values]
}
```

**Response:**
```json
{
  "modelVersion": 1,
  "denoised_signal": [array of 1300 float values]
}
```

## 🛠️ Tech Stack

- **Backend**: Python 3.12.0 or below, Flask 2.3.3, PostgreSQL, SQLAlchemy
- **Deep Learning**: PyTorch 2.6.0
- **Signal Processing**: NumPy

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/heartify-healthcare/heartify-denoised-model.git
cd heartify-denoised-model

# Install dependencies
pip install -r requirements.txt

# Configure .env file with your settings
# DATABASE_URL, SMTP_*, SECRET_KEY, ECG_MODEL_PATH, MODEL_VERSION, etc.

# Run server
python wsgi.py
```

## 🐳 Docker

```bash
docker-compose up -d
```

## 🔬 Model Details

**Attention U-Net**
- Architecture: Encoder-Decoder with Attention Gates
- Input: 1-lead ECG, 1300 samples (130Hz, 10 seconds)
- Output: Denoised ECG signal (same dimensions)
- Preprocessing: Z-score normalization
- Weights: `model/best_attention_unet.pth`

### Architecture Components
- **Encoder**: 4 ConvBlocks with MaxPooling (1→32→64→128→256 channels)
- **Center**: ConvBlock (256→512 channels)
- **Decoder**: 4 upsampling stages with Attention Gates
- **Output**: Conv1d layer (32→1 channel)

## 📚 Academic Context

This API was developed as part of a university **graduation thesis**, under the topic:

> **"Heart disease risk prediction using ECG signals with deep learning and large language models."**

## ✍️ Author

- [Vo Tran Phi](https://github.com/votranphi)
- [Le Duong Minh Phuc](https://github.com/minhphuc2544)

## 📄 License

This project is available under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) license.
