# ==========================================
# STAGE 1: BUILDER (Dùng để cài đặt và tải resource)
# ==========================================
FROM python:3.12-slim AS builder

WORKDIR /app

# 1. Cài đặt các công cụ build (chỉ cần ở bước này)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    python3-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. Tạo Virtual Environment để gom gọn thư viện
RUN python -m venv /opt/venv
# Kích hoạt venv cho các lệnh tiếp theo
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .

# 1. Cài đặt PyTorch bản CPU trước (khoảng 150MB thay vì 2-4GB của bản GPU)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 2. Xóa các dòng torch trong requirements.txt (để tránh pip tải lại bản GPU đè lên)
# Dùng sed để xóa các dòng chứa chữ 'torch'
RUN sed -i '/torch/d' requirements.txt

# 3. Cài các thư viện còn lại
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir huggingface_hub python-dotenv

# 4. Tải Model từ Hugging Face
ARG HF_REPO_ID="minhphuc2544/denoised-model"
ARG HF_FILENAME="best_attention_unet.pth"

RUN mkdir -p model
RUN python -c "from huggingface_hub import hf_hub_download; \
    hf_hub_download( \
        repo_id='${HF_REPO_ID}', \
        filename='${HF_FILENAME}', \
        local_dir='model', \
        local_dir_use_symlinks=False, \
    )"

# ==========================================
# STAGE 2: RUNNER (Image Production thật sự)
# ==========================================
FROM python:3.12-slim AS runner

WORKDIR /app

# Tối ưu hóa Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# 1. Cài đặt thư viện Runtime hệ thống (Nhẹ hơn nhiều so với dev)
# - libpq5: Cần thiết để chạy psycopg2 (PostgreSQL) mà không cần gcc/libpq-dev
# - libgomp1: Cần cho PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy Virtual Environment từ Builder sang
COPY --from=builder /opt/venv /opt/venv

# 3. Copy Model từ Builder sang
COPY --from=builder /app/model ./model

# 4. Copy Source Code & Config
COPY wsgi.py .
COPY app ./app
COPY heartify-denoised-model.env .

# 5. Security & Run
RUN adduser --disabled-password --gecos "" aiuser
USER aiuser

EXPOSE 5000

CMD ["python", "wsgi.py"]