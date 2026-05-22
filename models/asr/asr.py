import logging
import torch
from transformers import pipeline

logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"ASR using device: {device}")

asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    device=device
)

def transcribe(audio_path):
    """Transcribe English speech from an audio file to text."""
    try:
        result = asr_pipeline(
            audio_path,
            generate_kwargs={
                "language": "english"
            }
        )
        text = result["text"].strip()

        if not text:
            logger.warning("ASR produced empty transcription")

        return text

    except Exception as e:
        logger.error(f"ASR transcription failed: {e}")
        raise RuntimeError(f"Speech recognition failed: {e}") from e