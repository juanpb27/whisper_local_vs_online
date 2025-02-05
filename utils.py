import time
from queue import Queue, Empty
from threading import Thread
from tempfile import NamedTemporaryFile
from pydub import AudioSegment
from config import Config

class TranscriptionService:

    def __init__(self, client_build, model, local: bool):
        """Inicializa el servicio de transcripción."""
        self.client_build = client_build
        self.current_model = model
        self.local = local
        self.total_segments = 0
        self.failed_segments = 0

    def distribute_segments(self, audio_path: str, queue_segments: Queue, segment_length_ms: int = 30 * 1000):
        """Divide el audio en segmentos y los coloca en queue_segments con su índice."""
        try:
            audio = AudioSegment.from_file(audio_path)
            self.total_segments = (len(audio) // segment_length_ms) + 1
            print(f"Duración total del audio: {len(audio)} ms, {self.total_segments} segmentos.")

            for idx, start in enumerate(range(0, len(audio), segment_length_ms)):
                end = min(start + segment_length_ms, len(audio))
                segment = audio[start:end]
                queue_segments.put((idx, segment))
                print(f"Segmento {idx} de audio creado: {start} ms - {end} ms")
        except Exception as e:
            print(f"Error en distribute_segments: {e}")

    def transcribe_segment(self, queue_segments: Queue, results_queue: Queue):
        """Procesa segmentos de audio en paralelo y coloca los resultados ordenados por índice."""
        while True:
            try:
                item = queue_segments.get(timeout=5)
                if item is None:
                    break  # Señal de terminación

                idx, segment = item  # Extraemos el índice y el segmento
                print(f"🔹 Transcribiendo segmento {idx} de {segment.duration_seconds} segundos...")

                # Si el segmento es muy corto, omitirlo
                if segment.duration_seconds < 0.1:
                    print(f"⚠️ Segmento {idx} demasiado corto, omitiendo...")
                    continue

                try:
                    with NamedTemporaryFile(delete=False, suffix=".wav") as segment_file:
                        # 🔹 Convertir a PCM WAV, mono, 16kHz
                        segment.export(segment_file.name, format="wav", parameters=["-ac", "1", "-ar", "16000"])

                        if self.local:
                            transcription_result = self.current_model.transcribe(segment_file.name)
                            transcription_text = transcription_result["text"]  # 🔹 Extraer solo el texto
                        else:
                            with open(segment_file.name, "rb") as audio_file:
                                transcription_text = self.client_build.audio.transcriptions.create(
                                    model=self.current_model,
                                    language="es",
                                    file=audio_file,
                                    response_format="text",
                                    prompt="'EDOF', '2+', '3+'"
                                )

                        results_queue.put((idx, transcription_text, segment.duration_seconds))

                except Exception as e:
                    print(f"❌ Error al transcribir segmento {idx}: {e}")
                    self.failed_segments += 1
                    continue

            except Empty:
                break

    def transcribe_audio(self, file_path):
        """Orquestador de transcripción"""
        queue_segments = Queue()
        results_queue = Queue()

        start_time = time.time()

        print(f"Usando {worker} workers para transcripción")
        producer_thread = Thread(target=self.distribute_segments, args=(file_path, queue_segments))
        producer_thread.start()

        threads_list = []
        worker = Config.get_workers()
        for _ in range(worker):  # 🔹 Usa todos los workers calculados
            t = Thread(target=self.transcribe_segment, args=(queue_segments, results_queue))
            threads_list.append(t)
            t.start()

        producer_thread.join()
        for _ in threads_list:
            queue_segments.put(None)
        for th in threads_list:
            th.join()

        transcription_results = {}

        while not results_queue.empty():
            idx, transcription_text, _ = results_queue.get()
            if transcription_text:
                transcription_results[idx] = transcription_text

        # 🔹 Unir solo los textos transcritos
        transcription_texts = [transcription_results[i] for i in sorted(transcription_results.keys())]
        transcription_text = " ".join(transcription_texts)

        end_time = time.time()
        print(f"Tiempo total de transcripción: {end_time - start_time:.2f} segundos")
        return transcription_text
