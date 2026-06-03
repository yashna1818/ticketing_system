import whisper
import warnings

# Suppress FP16 warnings if CPU
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# Load a small base Whisper model to keep it lightweight but accurate
def get_whisper_model():
    print("Loading Whisper model (this may take a moment on first run)...")
    return whisper.load_model("base")

model = get_whisper_model()

def transcribe_audio(audio_path):
    """
    Takes an audio file path (e.g., WAV, MP3) and returns the transcribed text.
    """
    result = model.transcribe(audio_path)
    return result["text"]
