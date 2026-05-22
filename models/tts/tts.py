import logging
import torch
from transformers import VitsModel
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"TTS using device: {device}")

model_name = "facebook/mms-tts-mal"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = VitsModel.from_pretrained(model_name).to(device)

def text_to_speech(text):
    """Convert Malayalam text to speech audio.

    Returns a (sample_rate, numpy_array) tuple compatible
    with Gradio's audio output component.
    """
    try:
        inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model(**inputs).waveform

        audio = output.cpu().numpy()[0]
        sample_rate = model.config.sampling_rate

        return (sample_rate, audio)

    except Exception as e:
        logger.error(f"TTS synthesis failed: {e}")
        raise RuntimeError(f"Text-to-speech failed: {e}") from e