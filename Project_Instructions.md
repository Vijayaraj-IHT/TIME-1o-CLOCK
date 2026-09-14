# SIH Project: Voice Activator Instructions

This document outlines the complete end-to-end process for managing your SIH custom keyword spotting model, including training, deleting models, and managing custom keywords.

---

## 1. How to Train the Model

The model uses a **Metric Learning (Contrastive Loss)** approach, meaning it's trained to map audio into embeddings rather than fixed classes. 

### Prerequisites:
Make sure your Python virtual environment is activated before running any scripts:
```powershell
.\.venv\Scripts\Activate.ps1
```

### Steps to Train:
1. **Download and Prepare Data:** If you haven't downloaded the datasets yet, run the data preparation scripts first.
   ```powershell
   python scripts\download_all_datasets.py
   python scripts\prepare_all_data.py
   ```
2. **Run the Training Script:** Train the metric learning embedding model using the `train_metric.py` script.
   ```powershell
   python src\training\train_metric.py
   ```
3. **Export and Quantize:** After training completes, export the model to a highly compressed INT8 TensorFlow Lite format so it fits on your ESP32.
   ```powershell
   python scripts\prepare_tflite.py
   ```

### Where the trained files are saved

After training and export, the main files are:

- `models\checkpoints\tiny_cnn_metric_best.keras`: best FP32 Tiny CNN encoder.
- `models\checkpoints\ds_cnn_metric_best.keras`: best FP32 DS-CNN encoder.
- `models\tflite\voice_activator_int8.tflite`: universal INT8 encoder used by the streaming detector and ESP32 firmware.
- `outputs\tflite\tflite_micro_model.h`: C array generated for TFLite Micro firmware.
- `src\deployment\esp32\keyword_prototype.h`: the currently enrolled keyword prototype.

The `.tflite` file is the deployable trained model. The keyword is not embedded as a fixed classification class; it is represented by the prototype generated during enrollment.

## 2. How to Run the Trained Model on Windows

Run these commands from the project root after activating the virtual environment.

### Run the offline demonstration

This uses the INT8 model, enrolls the sample `ZORA` recordings, and runs a simulated continuous stream:

```powershell
python scripts\demo_voice_activator.py
```

The telemetry report is written to `experiments\sih_final_demo_report.json`.

### Run live microphone inference

First enroll a keyword, then start the real-time detector:

```powershell
python scripts\enroll_keyword.py --keyword AGNI --mic --shots 3
python scripts\live_mic_activator.py --keyword AGNI --threshold 0.88
```

Use `Ctrl+C` to stop the live detector. The microphone must be available to Python through `sounddevice`.

---

## 3. How to Delete the Model

If you want to clear out your trained models to start fresh (for instance, to retrain from scratch without loading previous checkpoints), you need to delete the saved weights and TFLite files.

### Steps to Delete:
Run the following commands in PowerShell from the project root (`D:\SIH_Model`):
```powershell
# 1. Delete all saved TensorFlow checkpoints and `.h5` model files:
Remove-Item -Recurse -Force D:\SIH_Model\models\checkpoints\*
Remove-Item -Recurse -Force D:\SIH_Model\models\saved_model\*

# 2. Delete the quantized TFLite binary files:
Remove-Item -Recurse -Force D:\SIH_Model\models\tflite\*.tflite
```
*Note: Make sure not to delete the actual `models` folder itself, just the contents inside it.*

---

## 4. How to Assign a New Keyword on the Development Computer

Because the system uses "Few-Shot Enrollment" and Metric Learning, **you do not need to retrain the model to add a new keyword.** You just need to extract a prototype for it.

### Steps to Assign:
Make sure your virtual environment is active, then run the enrollment script. 
You can use your laptop microphone to record the new keyword (Recommended):
```powershell
python scripts\enroll_keyword.py --keyword YOUR_NEW_KEYWORD --mic
```
*Alternatively, if you don't have a mic, use Windows Text-to-Speech to synthesize it:*
```powershell
python scripts\enroll_keyword.py --keyword YOUR_NEW_KEYWORD --synth
```
This script automatically generates the new C++ `keyword_prototype.h` file for your ESP32-S3.

---

## 5. How to Delete the Pretrained Keyword and Assign a New One

If you want to wipe the default/pretrained keyword (e.g., `"zora"`) completely before adding yours (especially useful before merging your code to keep the repository clean), follow these steps:

### Step 1: Delete the old keyword data and C++ headers
Run these commands in PowerShell to delete the `"zora"` files:
```powershell
# Delete the audio samples for the old keyword
Remove-Item -Recurse -Force D:\SIH_Model\data\raw\custom_keywords\zora

# Delete the auto-generated C++ headers deployed to the ESP32 source folders
Remove-Item -Path "D:\SIH_Model\src\deployment\esp32\keyword_prototype.h" -ErrorAction SilentlyContinue
Remove-Item -Path "D:\SIH_Model\src\deployment\esp32_wroom\keyword_prototype.h" -ErrorAction SilentlyContinue
Remove-Item -Path "D:\SIH_Model\outputs\esp32_wroom\keyword_prototype.h" -ErrorAction SilentlyContinue
```

### Step 2: Assign the new keyword
Now that the system is clean, assign your new keyword (e.g., `"AGNI"`) using the microphone:
```powershell
python scripts\enroll_keyword.py --keyword AGNI --mic
```
The script will freshly generate the necessary `.h` headers for the ESP32 deployment folders automatically.

---

## 6. Running the Model on a Raspberry Pi

There are two different Raspberry Pi use cases:

1. **Pi as a host prototype:** Python reads the Pi microphone, runs the `.tflite` encoder, and compares embeddings with a keyword prototype.
2. **Pi as a controller for the ESP32:** the ESP32 performs low-latency microphone inference and enrollment, while the Pi receives the wake signal or ASR audio.

The repository currently provides complete firmware support for the second architecture in `src\deployment\esp32_wroom`. It does not yet provide a Raspberry Pi-specific service or GPIO/audio capture application, and the Python scripts currently contain the Windows project path `D:\SIH_Model`. Therefore, do not copy the Windows commands to a Pi unchanged.

### Raspberry Pi host-prototype setup

On the Pi, copy or clone the project, then use Linux paths and a Python virtual environment:

```bash
cd ~/SIH_Model
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For live microphone input, install the PortAudio development package before installing `sounddevice`:

```bash
sudo apt update
sudo apt install -y portaudio19-dev libsndfile1
python -m pip install sounddevice soundfile
```

Before running the Python detector, update the hard-coded `D:\SIH_Model` project-root values in these scripts to the Pi checkout path (for example `/home/pi/SIH_Model`):

- `scripts\live_mic_activator.py`
- `scripts\enroll_keyword.py`
- `src\streaming\demo_pipeline.py`

Use the existing INT8 model and a directory of Pi-recorded WAV files to enroll a keyword:

```bash
python scripts/enroll_keyword.py --keyword AGNI --audio-dir /home/pi/agni_samples --shots 3 --no-purge
python scripts/live_mic_activator.py --keyword AGNI --threshold 0.88
```

The WAV files should be clear, single-channel recordings at 16 kHz, with at least three utterances recommended. `--audio-dir` is the Pi-compatible enrollment mode; `--synth` is Windows-only because it uses Windows SAPI. Enrollment writes the prototype header, after which the live detector uses the same frozen encoder without retraining.

> **Compatibility note:** TensorFlow installation on Raspberry Pi depends on the Pi OS, Python version, and CPU architecture. If `pip install -r requirements.txt` cannot install TensorFlow, use a compatible TensorFlow Lite runtime and adapt the imports in the Python inference scripts from `tensorflow.lite.Interpreter` to that runtime. Validate the model input shape `(1, 98, 13, 1)` and output shape `(1, 32)` before live testing.

### Recommended Raspberry Pi plus ESP32 deployment

For the intended low-latency design, flash the firmware in `src\deployment\esp32_wroom` to the ESP32-WROOM with the INT8 model header and microphone connected. The Raspberry Pi can then act as the host for ASR or application logic. The ESP32 performs detection locally and sends the wake pulse on GPIO 4.

This route already supports adding a new keyword directly on the device:

1. Connect the INMP441 microphone according to `outputs\esp32_wroom\README.md`.
2. Flash `src\deployment\esp32_wroom\voice_activator_esp32_wroom.ino` once.
3. Hold the ESP32 BOOT button on GPIO 0 for about two seconds.
4. Speak the new keyword three times when the blue LED signals recording.
5. The ESP32 computes the 32-D prototype and stores it in flash NVS.
6. Rebooting does not erase the keyword; the stored prototype is loaded automatically.

No model retraining or Raspberry Pi connection is needed to change the keyword in this mode. Hold BOOT for about five seconds to restore the firmware-default keyword. Follow the complete wiring, library, board, and upload instructions in `outputs\esp32_wroom\README.md`.

### Raspberry Pi deployment limitation

The Pi host-prototype path currently enrolls from WAV files or from a Python-accessible microphone and stores the generated prototype as a header on disk. It does not yet save prototypes in a Pi service database or expose a button/API for enrollment. For persistent, standalone on-device keyword enrollment today, use the ESP32-WROOM firmware path above.

---

## 7. Self-Contained Raspberry Pi Package

The folder `raspberry_pi_deployment` is a small portable package containing the trained INT8 TFLite model, the runner, keyword enrollment, and keyword storage:

```text
raspberry_pi_deployment/
├── voice_activator_int8.tflite
├── model_runtime.py
├── run_model.py
├── add_keyword.py
└── custom_keywords/
```

Copy this complete folder to the Raspberry Pi. Install the package dependencies from the project requirements, then run from inside the folder:

```bash
cd ~/raspberry_pi_deployment
python add_keyword.py --keyword AGNI --audio-dir /home/pi/agni_samples --shots 3
python run_model.py --mic --threshold 0.88
```

For existing WAV recordings instead of the Pi microphone:

```bash
python add_keyword.py --keyword AGNI --audio-dir /home/pi/agni_samples --shots 3
python run_model.py --wav /home/pi/agni_samples/agni_1.wav
```

The enrollment script stores recordings under `custom_keywords/AGNI/` and writes the active 32-D prototype to `custom_keywords/current_keyword.json`. Running `add_keyword.py` again replaces the active keyword without retraining the TFLite model.

## 8. Raspberry Pi Without a Microphone: Laptop Microphone Streaming

If the Raspberry Pi has no microphone, use the laptop microphone as the audio source. Connect both devices to the same hotspot Wi-Fi network, with SSID `////`. Enter the hotspot password directly in the laptop and Pi Wi-Fi settings; the password is intentionally not saved in source code or project files.

The package now contains:

- `pi_stream_server.py`: runs on the Pi, receives audio, and runs the trained model.
- `laptop_mic_client.py`: runs on the laptop, captures the laptop microphone, and streams 16 kHz audio to the Pi.
- `requirements.txt`: Python libraries needed by the package.
- `README.md`: the complete deployment guide.

### Install libraries on the Raspberry Pi

Copy `raspberry_pi_deployment` to the Pi and run:

```bash
cd ~/raspberry_pi_deployment
sudo apt update
sudo apt install -y python3-venv libsndfile1
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The Pi requires `tflite-runtime`, NumPy, SciPy, librosa, and soundfile. `sounddevice` is listed for the laptop client; it is not required on the Pi because the Pi has no microphone. If no `tflite-runtime` wheel exists for the Pi's Python version and architecture, install a compatible wheel before starting the server. Do not install full TensorFlow on the Pi.

### Start the Pi server

Find the Pi hotspot address:

```bash
hostname -I
```

Start the model server on the Pi:

```bash
source .venv/bin/activate
python pi_stream_server.py --threshold 0.88
```

The server listens on TCP port `8765` and waits for the laptop.

### Start the laptop microphone client

Install the laptop-side audio libraries from inside the copied package folder:

```powershell
python -m pip install numpy sounddevice
```

Then replace `PI_IP_ADDRESS` with the address printed by `hostname -I`:

```powershell
python laptop_mic_client.py --pi-host PI_IP_ADDRESS
```

The laptop sends 50 ms audio frames over the hotspot. The Pi maintains a one-second rolling window, extracts MFCC features, runs the H5 model, compares the embedding with the active keyword prototype, and prints `DETECTED` when the threshold is crossed. Press `Ctrl+C` on the laptop to stop.
