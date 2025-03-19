# **NanoDet-Based Object Detection Project**

## **📌 Project Overview**

This project integrates **NanoDet**, a lightweight object detection model, with **Vosk**, an offline speech recognition toolkit. The system allows real-time detection of objects in images and videos, making it ideal for embedded AI applications like **IoT devices** and **assistive technology**.

---

## **⚙️ Dependencies & Libraries**

To run this project, install the following dependencies:

### **🖥 Core Libraries**

- `torch` – Deep learning framework for running NanoDet.
- `torchvision` – Image processing utilities for PyTorch.
- `onnxruntime` – For ONNX model inference.
- `numpy` – Efficient numerical computing.
- `opencv-python` – Image processing & visualization.
- `pyyaml` – Configuration file handling.

### **🎤 Speech Recognition (Vosk)**

- `vosk` – Offline speech recognition.
- `sounddevice` – Access microphone input.
- `wave` – Handle WAV audio files.

### **📦 Additional Utilities**

- `argparse` – Command-line argument parsing.
- `logging` – Debugging and logging events.
- `matplotlib` – Optional visualization library.

---

## **🔧 Installation**

Follow these steps to install all dependencies:

### **1️⃣ Install Python Packages**

Run the following command to install all required libraries:

```sh
pip install torch torchvision numpy opencv-python pyyaml onnxruntime vosk sounddevice wave argparse logging
```

### **2️⃣ Download and Setup NanoDet**

```sh
git clone https://github.com/RangiLyu/nanodet.git
cd nanodet
pip install -r requirements.txt
```

### **3️⃣ Download Pretrained NanoDet Model**

```sh
mkdir model
wget https://github.com/RangiLyu/nanodet/releases/download/v0.4.0/nanodet_m.pth -O model/nanodet_m.pth
```

### **4️⃣ Download a Vosk Model**

1. Visit: [Vosk Models](https://alphacephei.com/vosk/models)
2. Download a language model (e.g., `vosk-model-en-us-0.22.zip`).
3. Extract it to a folder, e.g., `C:/vosk_model/`.

---

## **🚀 Running the Project**

### **🔹 1. Run Object Detection on Images**

```sh
python demo/demo.py image --config config/nanodet-m.yml --model model/nanodet_m.pth --path test.jpg
```

### **🔹 2. Run Object Detection on Video**

```sh
python demo/demo.py video --config config/nanodet-m.yml --model model/nanodet_m.pth --path test.mp4
```

### **🔹 3. Run Real-Time Speech Recognition**

```sh
python speech_recognition.py
```

---

## **🐛 Troubleshooting**

- **CUDA Not Found?** Run on CPU instead:
  ```sh
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  ```
- **NumPy 2.x Error?** Downgrade:
  ```sh
  pip install numpy<2.0
  ```
- **Git Issues?** Ensure you're on the correct branch:
  ```sh
  git checkout main
  ```

---

## **📜 License**

This project is open-source under the **MIT License**.

---

## **📞 Contact & Support**

For questions, contact: [Your Email or GitHub Issues].

