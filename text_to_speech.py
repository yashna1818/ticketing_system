from gtts import gTTS
import tempfile

def synthesize_speech(text):
    """
    Converts a text string directly into spoken human-like audio using Google TTS.
    Returns the absolute path to the generated .mp3 file.
    """
    tts = gTTS(text=text, lang='en', slow=False)
    temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_audio_file.name)
    return temp_audio_file.name
