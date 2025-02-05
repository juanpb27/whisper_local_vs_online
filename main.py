import argparse
import os
from utils import TranscriptionService
from config import Config
import torch

def download_model():
    """Descarga el modelo de Whisper si no está ya en la carpeta"""
    if not os.path.exists(Config.MODEL_PATH):
        os.makedirs(Config.MODEL_PATH, exist_ok=True)

    print(f"Descargando modelo Whisper ({Config.LOCAL_MODEL_NAME}) en {Config.MODEL_PATH}...")
    try:
        import whisper
        if torch.cuda.is_available():
            print(f"✅ GPU detectada: {torch.cuda.get_device_name(0)}")
            torch.cuda.set_device(0)  # 🔹 Asegura que usa la GPU
        else:
            print("❌ No se detectó GPU, se usará CPU.")

        whisper.load_model(Config.LOCAL_MODEL_NAME, download_root=Config.MODEL_PATH)
        print("✅ Modelo descargado correctamente.")
    except Exception as e:
        print(f"❌ Error descargando el modelo: {e}")

def transcribe_audio(local: bool):
    """Transcribe el audio usando Whisper local o la API en la nube"""
    if not os.path.exists(Config.AUDIO_PATH):
        print(f"❌ Error: El archivo de audio {Config.AUDIO_PATH} no existe.")
        return

    client, model = Config.get_client_and_model(local)

    if client is None and model is None:
        return  # Error ya mostrado en `get_client_and_model`

    transcription_service = TranscriptionService(client, model, local)
    transcription_text = transcription_service.transcribe_audio(Config.AUDIO_PATH)

    print("\n🔹 Transcripción final:\n", transcription_text)

def main():
    parser = argparse.ArgumentParser(description="Herramienta de transcripción de audio con Whisper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("download", help="Descargar el modelo de Whisper localmente")

    transcribe_parser = subparsers.add_parser("transcribe", help="Transcribir un audio")
    transcribe_parser.add_argument("--local", action="store_true", help="Usar Whisper local en lugar de la API en la nube")

    args = parser.parse_args()

    if args.command == "download":
        download_model()
    elif args.command == "transcribe":
        transcribe_audio(args.local)

if __name__ == "__main__":
    main()
