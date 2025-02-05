import os
import whisper
import torch
from dotenv import load_dotenv
from openai import OpenAI

# Cargar variables de entorno
load_dotenv()

class Config:
    MODEL_PATH = "whisper_model"
    LOCAL_MODEL_NAME = "turbo"
    FIREWORKS_MODEL_NAME = "whisper-v3"
    AUDIO_PATH = "audio.mp3"
    FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")

    @staticmethod
    def get_device():
        """Devuelve 'cuda' si hay GPU disponible, de lo contrario 'cpu'."""
        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def get_client_and_model(local: bool):
        """Inicializa y retorna el modelo de Whisper local o el cliente de Fireworks AI."""
        if local:
            try:
                device = Config.get_device()
                print(f"🔹 Cargando modelo local {Config.LOCAL_MODEL_NAME} en {device}...")

                model = whisper.load_model(Config.LOCAL_MODEL_NAME, download_root=Config.MODEL_PATH)
                model = model.to(device)  # Mover modelo a GPU si está disponible

                print(f"✅ Modelo local {Config.LOCAL_MODEL_NAME} cargado en {device}.")
                return None, model
            except Exception as e:
                print(f"❌ Error cargando modelo local {Config.LOCAL_MODEL_NAME}: {e}")
                return None, None

        else:
            if not Config.FIREWORKS_API_KEY:
                print("❌ Error: No se encontró la API Key de Fireworks AI en el .env")
                return None, None

            print(f"🔹 Usando Fireworks AI con modelo {Config.FIREWORKS_MODEL_NAME}...")
            client = OpenAI(
                api_key=Config.FIREWORKS_API_KEY,
                base_url="https://api.fireworks.ai/inference/v1"
            )
            return client, None

    @staticmethod
    def get_workers():
        """Calcula el número óptimo de workers en función de CPU o GPU."""
        total_cores = os.cpu_count() or 4  # Si no se detectan, asumimos 4 cores
        device = Config.get_device()

        if device == "cuda":
            # 🔹 Para GPU, usar más workers (máximo 8)
            return min(8, total_cores // 2)
        else:
            # 🔹 Para CPU, usa menos workers (máximo 4)
            return min(4, total_cores // 4)