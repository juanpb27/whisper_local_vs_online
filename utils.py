import time
import threading
import multiprocessing
import torch
from tempfile import NamedTemporaryFile
from pydub import AudioSegment
from config import Config

multiprocessing.set_start_method("spawn", force=True)  # 🔹 Forzar spawn para evitar el error de CUDA

class TranscriptionService:
    def __init__(self, client_build, model, local: bool):
        self.client_build = client_build
        self.current_model = model
        self.local = local
        self.total_segments = 0
        self.failed_segments = 0

    def distribute_segments(self, audio_path: str, segment_length_ms: int = 30 * 1000):
        try:
            audio = AudioSegment.from_file(audio_path)
            self.total_segments = (len(audio) // segment_length_ms) + 1
            print(f"Duración total del audio: {len(audio)} ms, {self.total_segments} segmentos.")

            segments = []
            for idx, start in enumerate(range(0, len(audio), segment_length_ms)):
                end = min(start + segment_length_ms, len(audio))
                segment = audio[start:end]
                segments.append((idx, segment))
                print(f"Segmento {idx} de audio creado: {start} ms - {end} ms")
            
            return segments
        except Exception as e:
            print(f"Error en distribute_segments: {e}")
            return []

    def transcribe_segment(self, item):
        try:
            idx, segment = item
            print(f"🔹 Transcribiendo segmento {idx} de {segment.duration_seconds} segundos...")

            if segment.duration_seconds < 0.1:
                print(f"⚠️ Segmento {idx} demasiado corto, omitiendo...")
                return (idx, "")

            with NamedTemporaryFile(delete=False, suffix=".wav") as segment_file:
                segment.export(segment_file.name, format="wav", parameters=["-ac", "1", "-ar", "16000"])

                if self.local:
                    transcription_result = self.current_model.transcribe(segment_file.name)
                    transcription_text = transcription_result["text"]
                    torch.cuda.empty_cache()  # 🔹 Libera memoria CUDA después de cada transcripción

                    # 🔹 Liberar memoria CUDA manualmente
                    torch.cuda.empty_cache()
                else:
                    with open(segment_file.name, "rb") as audio_file:
                        transcription_text = self.client_build.audio.transcriptions.create(
                            model=self.current_model,
                            language="es",
                            file=audio_file,
                            response_format="text"
                        )

                return (idx, transcription_text)

        except Exception as e:
            print(f"❌ Error al transcribir segmento {idx}: {e}")
            return (idx, "")

    def transcribe_audio(self, file_path):
        start_time = time.time()
        segments = self.distribute_segments(file_path)

        worker_count = Config.get_workers()
        print(f"🖥️ Usando {worker_count} procesos para transcripción.")

        results = {}

        if self.local:
    # 🔹 Multiprocessing para optimizar ejecución en GPU
            with multiprocessing.Pool(worker_count) as pool:
                results_list = pool.map(self.transcribe_segment, segments)
                results = {idx: text for idx, text in results_list if text}
        else:
            # 🔹 Threading para Fireworks AI (evita problemas con serialización)
            threads = []
            results_lock = threading.Lock()

            def worker(segment):
                idx, text = self.transcribe_segment(segment)
                with results_lock:
                    results[idx] = text

            for segment in segments:
                thread = threading.Thread(target=worker, args=(segment,))
                thread.start()
                threads.append(thread)

            for thread in threads:
                thread.join()

        # 🔹 Ordenar resultados antes de concatenarlos
        transcription_texts = [results[i] for i in sorted(results.keys())]
        transcription_text = " ".join(transcription_texts)

        end_time = time.time()
        print(f"Tiempo total de transcripción: {end_time - start_time:.2f} segundos")
        return transcription_text
