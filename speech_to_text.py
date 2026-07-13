import whisper
import warnings

# Suppress FP16 warnings if CPU
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

_model_cache = {}

def get_whisper_model(model_name="base"):
    """
    Lazily loads and caches the Whisper model to avoid import-time overhead.
    """
    if model_name not in _model_cache:
        print(f"Loading Whisper model '{model_name}' (this may take a moment on first run)...")
        _model_cache[model_name] = whisper.load_model(model_name)
    return _model_cache[model_name]

def transcribe_audio(audio_path, model_name="base", api_key=None, language=None):
    """
    Takes an audio file path (e.g., WAV, MP3) and returns the transcribed text.
    Uses OpenAI API if api_key is provided, falling back to local Whisper on failure.
    """
    if api_key and api_key.strip():
        import requests
        print(f"API Key detected. Attempting OpenAI cloud transcription in language={language}...")
        try:
            headers = {
                "Authorization": f"Bearer {api_key}"
            }
            with open(audio_path, "rb") as f:
                files = {
                    "file": f,
                    "model": (None, "whisper-1")
                }
                data = {}
                if language:
                    data["language"] = language
                response = requests.post(
                    "https://api.openai.com/v1/audio/transcriptions", 
                    headers=headers, 
                    files=files, 
                    data=data
                )
            if response.status_code == 200:
                return response.json()["text"]
            else:
                print(f"OpenAI API error: {response.text}. Falling back to local Whisper.")
        except Exception as e:
            print(f"Failed to transcribe via API: {e}. Falling back to local Whisper.")

    model = get_whisper_model(model_name)
    # Whisper expects 2-letter ISO code or None
    result = model.transcribe(audio_path, language=language) if language else model.transcribe(audio_path)
    return result["text"]


