import os
import sys
from dotenv import load_dotenv

# --- DEBUG BLOCK ---
print("--- [BOOT] Starting wsgi.py ---")
env_file = 'heartify-denoised-model.env'

if os.path.exists(env_file):
    print(f"--- [BOOT] Found env file: {env_file}")
    load_dotenv(env_file, override=True)
else:
    print(f"--- [BOOT] WARNING: {env_file} not found! Listing current dir:")
    print(os.listdir('.'))
    load_dotenv() # Fallback to .env

# --- APP IMPORT ---
from app import create_app

# Tạo ứng dụng Flask
app = create_app()

if __name__ == "__main__":
    # Lấy port từ biến môi trường hoặc mặc định là 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
