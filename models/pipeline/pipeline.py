import logging

from models.asr.asr import transcribe
from models.translation.translation import translate_to_malayalam
from models.tts.tts import text_to_speech

logger = logging.getLogger(__name__)

def run_pipeline(audio_path):
    """Run the full STTS pipeline: ASR → Translation → TTS.

    Args:
        audio_path: Path to the input English audio file.

    Returns:
        Tuple of (english_text, malayalam_text, audio_output)
        where audio_output is a (sample_rate, numpy_array) tuple.
    """

    # Step 1: English speech → English text
    logger.info("Step 1/3: Transcribing English audio...")
    english_text = transcribe(audio_path)
    logger.info(f"ASR output: {english_text}")

    # Step 2: English text → Malayalam text
    logger.info("Step 2/3: Translating to Malayalam...")
    malayalam_text = translate_to_malayalam(english_text)
    logger.info(f"Translation output: {malayalam_text}")

    # Step 3: Malayalam text → Malayalam speech
    logger.info("Step 3/3: Synthesizing Malayalam speech...")
    audio_output = text_to_speech(malayalam_text)
    logger.info("Pipeline complete.")

    return (
        english_text,
        malayalam_text,
        audio_output
    )