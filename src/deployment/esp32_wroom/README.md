# ESP32-WROOM Low-Latency Voice Activator Deployment Guide
**SIH Problem Statement 26172 (ISRO)**

Complete guide and source files to run the trained INT8 Voice Activator on an **ESP32-WROOM-32** (ESP32 DevKit v1 / NodeMCU-32S) using standard **Arduino IDE** or **PlatformIO**.

---

## 🌟 Key Feature: On-Device Keyword Enrollment
**You can change the custom keyword directly on the ESP32 using only the onboard BOOT button and microphone!**
- No computer or USB connection needed.
- No re-flashing or model retraining.
- Keywords are saved into the ESP32's **Flash NVS memory** and persist across power cycles.

---

## 1. Hardware Specifications & Pinout

### Required Hardware
1. **ESP32-WROOM-32** Dev Module (Xtensa Dual-Core LX6 @ 240 MHz, 520 KB SRAM, 4MB Flash)
2. **INMP441** I2S Digital MEMS Microphone (or ICS-43434 / SPH0645)
3. Micro-USB cable & breadboard jumper wires

### Wiring Diagram: INMP441 to ESP32-WROOM-32

| INMP441 Microphone Pin | ESP32-WROOM Pin | Description |
| :--- | :--- | :--- |
| **VDD** | **3V3** | 3.3V Power |
| **GND** | **GND** | Ground |
| **SD** (Serial Data) | **GPIO 32** | I2S Data In (`I2S_SD_PIN`) |
| **SCK** (Serial Clock) | **GPIO 14** | I2S Bit Clock (`I2S_SCK_PIN`) |
| **WS** (Word Select) | **GPIO 15** | I2S Word Select / LR Clock (`I2S_WS_PIN`) |
| **L/R** (Channel) | **GND** | Connect to GND for Left Channel Mono |

### Onboard Buttons & Indicators
- **GPIO 0 (BOOT Button)**: 
  - **Hold for 2 seconds**: Enters **On-Device Keyword Enrollment Mode**.
  - **Hold for 5 seconds**: Resets custom keyword back to firmware default.
- **GPIO 2 (Blue LED)**:
  - **Solid ON**: "Speak your keyword now!" (During enrollment).
  - **Double Blink**: Shot recorded successfully.
  - **Lit for 1.5s**: Keyword detected during live listening.
- **GPIO 4**: High-level wake pulse (Connect to host MCU to trigger ASR handover).

---

## 2. How to Change Keyword Directly on the ESP32 (No PC Needed!)

1. Power on the ESP32 (via USB, battery pack, or 5V pin).
2. **Press and hold the onboard BOOT button for 2 seconds** until the blue LED turns ON, then release it.
   *(Alternatively, if connected to Serial Monitor, type `enroll` and press Enter).*
3. The ESP32 enters Enrollment Mode:
   - **Shot 1**: When the blue LED turns **solid ON**, speak your new keyword clearly (e.g. *"JARVIS"*). The LED blinks twice when recorded.
   - **Shot 2**: When the blue LED turns **solid ON** again, speak the keyword a second time.
   - **Shot 3**: When the blue LED turns **solid ON** again, speak the keyword a third time.
4. The ESP32 computes the new 32-D centroid vector, validates unit norm, and writes it to **Flash NVS**.
5. The blue LED flashes **6 rapid pulses**: Enrollment is complete!
6. The ESP32 immediately starts listening for your new keyword!

> **Note**: Your new keyword is saved in non-volatile flash. It **remains active even after unplugging or rebooting** the board.

---

## 3. Deployment via Arduino IDE (Initial Setup)

### Step 1: Install Arduino Board & Library
1. Open **Arduino IDE** (v2.x recommended).
2. Go to **File -> Preferences**, and paste into *Additional Board Manager URLs*:
   ```
   https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
   ```
3. Go to **Tools -> Board -> Boards Manager**, search for `esp32` by **Espressif Systems**, and install it.
4. Go to **Sketch -> Include Library -> Manage Libraries**, search for `TensorFlowLite_ESP32` by **Tanaka Masayuki**, and click **Install**.

### Step 2: Open the Sketch Folder
Open `voice_activator_esp32_wroom.ino` from this folder in Arduino IDE.

### Step 3: Configure Board Settings
In the Arduino IDE **Tools** menu:
- **Board**: `ESP32 Dev Module`
- **CPU Frequency**: `240MHz (WiFi/BT)`
- **Flash Frequency**: `80MHz`
- **Flash Mode**: `QIO`
- **Partition Scheme**: `Default 4MB with spiffs (1.2MB APP / 1.5MB SPIFFS)`
- **Port**: Select your ESP32 COM port (e.g. `COM3` / `COM4`)

### Step 4: Compile & Upload
1. Click **Upload** (Arrow icon).
2. Open **Tools -> Serial Monitor** at **115200 baud**.
3. Speak your keyword into the INMP441 microphone to trigger activation!

---

## 4. Memory Footprint on ESP32-WROOM

| Memory Type | Required by Voice Activator | Total Available on ESP32-WROOM | Headroom Remaining |
| :--- | :--- | :--- | :--- |
| **Flash** | **56.4 KB** (TFLite INT8 model) | 4,096 KB (4 MB) | **98.6% free** |
| **Internal SRAM** | **64.0 KB** (Tensor Arena) + **31.3 KB** (Ring Buffer) = **95.3 KB** | ~320 KB (Free internal heap) | **~224 KB free** |
| **NVS Storage** | **128 bytes** (32 floats prototype) | 20 KB default NVS partition | **99.3% free** |
| **Inference Time** | **~35 - 45 ms** | 50 ms streaming chunk | **Real-time 1.0x capable** |
