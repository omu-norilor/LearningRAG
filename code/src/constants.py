import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")  # → code/.env
HF_TOKEN = os.getenv("HF_TOKEN")