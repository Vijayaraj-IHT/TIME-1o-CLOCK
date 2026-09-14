# Laptop Microphone + Raspberry Pi INT8 Inference

The Raspberry Pi runs the trained INT8 TFLite model. The laptop supplies microphone audio over the hotspot. No microphone is required on the Pi.

The package intentionally does not contain the training H5 file or TensorFlow. It contains the 57.76 KB `voice_activator_int8.tflite` model and uses `tflite-runtime`.

## 1. Connect both devices

Connect the laptop and Raspberry Pi to the same hotspot Wi-Fi network. The hotspot SSID is the one configured for this project. Enter the hotspot password locally in the operating system Wi-Fi dialog; it is intentionally not stored in this project.

Find the Pi address on the Pi:

```bash
hostname -I
```

Use the first address, for example `192.168.43.120`.

## 2. Install libraries on the Pi

Copy this complete folder to the Pi, then run:

```bash
cd ~/raspberry_pi_deployment
sudo apt update
sudo apt install -y python3-venv libsndfile1
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `tflite-runtime` has no wheel for your Pi's Python version, install a compatible `tflite-runtime` wheel for the Pi OS architecture before running the server. Do not install full TensorFlow on the Pi.

## 3. Enroll a keyword

The Pi can enroll from WAV files because it has no microphone. Copy at least three 16 kHz mono WAV recordings to the Pi:

```bash
python add_keyword.py --keyword AGNI --audio-dir /home/pi/agni_samples --shots 3
```

The active keyword is saved in `custom_keywords/current_keyword.json`. Recordings are kept under `custom_keywords/AGNI/` when created by the package.

## 4. Start the Pi model server

```bash
source .venv/bin/activate
python pi_stream_server.py --threshold 0.88 --interval-frames 2 --persistence 2
```

Leave this terminal running. The server listens on TCP port `8765` and prints similarity scores and detections.

## 5. Start the laptop microphone client

Install the laptop-side dependencies in the same package folder:

```powershell
python -m pip install numpy sounddevice
```

Then run this on the laptop, replacing the address with the Pi address from `hostname -I`:

```powershell
python laptop_mic_client.py --pi-host 192.168.43.120
```

The laptop captures 16 kHz audio in 50 ms frames and sends it to the Pi. The Pi performs feature extraction, model inference, and keyword comparison. Press `Ctrl+C` on the laptop to stop streaming.

The server skips low-energy frames and runs one inference every two frames. This reduces average CPU usage compared with running inference on every 50 ms frame. Actual CPU and RAM usage depend on the Pi model, OS, TensorFlow Lite backend, and other running services. The 256 KB RAM target applies to the embedded inference allocation, not total Raspberry Pi system memory; a Pi showing 610 MB total OS memory usage cannot meet a 256 KB whole-system target.
