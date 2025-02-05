# Prueba de Concepto para comparar Whisper local vs online en diferentes máquinas

## 📌 Comandos:

- **Descarga el modelo una vez:**
  ```bash
  python main.py download
  ```

- **Transcribe usando Whisper local:**
  ```bash
  python main.py transcribe --local
  ```

- **Transcribe usando la API en la nube:**
  ```bash
  python main.py transcribe
  ```

---

## 📌 Instalación y configuración de `torch`
### 1️⃣ Verificar la versión de CUDA instalada
Para saber qué versión de CUDA tienes en tu máquina o en RunPod, ejecuta en la terminal:
```bash
nvcc --version
```
Si **CUDA está instalada**, verás algo como:
```plaintext
nvcc: NVIDIA (R) Cuda compiler
Built on Sun_Jul_23_19:09:12_PDT_2023
Cuda compilation tools, release 12.1, V12.1.105
```
📌 **En este caso, la versión de CUDA es `12.1`.**

### 2️⃣ Verificar si PyTorch reconoce la GPU
Ejecuta en Python:
```python
import torch
print("CUDA disponible:", torch.cuda.is_available())
print("Versión de CUDA detectada por PyTorch:", torch.version.cuda)
print("Nombre de la GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No hay GPU")
```
Si **PyTorch detecta la GPU correctamente**, verás algo como:
```plaintext
CUDA disponible: True
Versión de CUDA detectada por PyTorch: 12.1
Nombre de la GPU: NVIDIA A100-SXM4-40GB
```
📌 **Si `torch.cuda.is_available()` devuelve `False`, entonces PyTorch no está configurado correctamente para usar la GPU.**

### 3️⃣ Instalación de PyTorch con la versión correcta de CUDA
Si `nvcc --version` muestra **CUDA 12.1**, instala PyTorch con:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
Si `nvcc --version` muestra **CUDA 11.8**, instala con:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Si **no tienes una GPU o quieres usar CPU**, instala PyTorch con:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

📌 **No es necesario instalar CUDA manualmente. PyTorch ya incluye los drivers correctos según la versión que instales.**

---

## 📌 Dependencias del proyecto
**Instala las demás dependencias del proyecto sin `torch`**:
```bash
pip install -r requirements.txt
```
Luego, **instala la versión correcta de `torch` según tu hardware (ver arriba).**

🚀 **Con esto, estarás listo para ejecutar las pruebas en Whisper local y en la nube.**
