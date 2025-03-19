import wave
import json
import vosk

# Path to Vosk model
MODEL_PATH = "C:./Vosk/vosk_model/vosk-model-ja-0.22"

# Load model
model = vosk.Model(MODEL_PATH)

# Open an audio file (must be WAV format, mono, 16-bit, 16kHz)
wf = wave.open("test_audio.wav", "rb")

if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
    print("Audio file must be WAV format, mono, 16-bit, 16kHz.")
    exit(1)

rec = vosk.KaldiRecognizer(model, wf.getframerate())

while True:
    data = wf.readframes(4000)
    if len(data) == 0:
        break
    if rec.AcceptWaveform(data):
        print(json.loads(rec.Result())["text"])

print(json.loads(rec.FinalResult())["text"])
