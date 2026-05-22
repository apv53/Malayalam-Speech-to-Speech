import logging
import gradio as gr

from models.pipeline.pipeline import run_pipeline

# Configure logging for all modules
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)

logger = logging.getLogger(__name__)

def infer(audio):
    """Gradio inference function for the STTS pipeline."""
    if audio is None:
        raise gr.Error("Please provide an audio input.")

    english_text, mal_text, out_audio = run_pipeline(audio)

    return (
        english_text,
        mal_text,
        out_audio
    )

demo = gr.Interface(
    fn=infer,
    inputs=gr.Audio(type="filepath"),
    outputs=[
        gr.Textbox(label="English Text"),
        gr.Textbox(label="Malayalam Translation"),
        gr.Audio(label="Malayalam Speech")
    ],
    title="Malayalam STTS",
    description="Upload English audio to get Malayalam text translation and speech output."
)

demo.launch(show_error=True)