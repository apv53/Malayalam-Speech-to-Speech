import logging
import torch
from transformers import MBartForConditionalGeneration
from transformers import MBart50TokenizerFast

logger = logging.getLogger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Translation using device: {device}")

model_name = "facebook/mbart-large-50-many-to-many-mmt"

tokenizer = MBart50TokenizerFast.from_pretrained(model_name)

model = MBartForConditionalGeneration.from_pretrained(
    model_name
).to(device)

def translate_to_malayalam(text):
    """Translate English text to Malayalam using mBART-50."""
    try:
        tokenizer.src_lang = "en_XX"

        encoded = tokenizer(
            text,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            generated_tokens = model.generate(
                **encoded,
                forced_bos_token_id=tokenizer.lang_code_to_id["ml_IN"]
            )

        translated = tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True
        )

        result = translated[0].strip()

        if not result:
            logger.warning("Translation produced empty output")

        return result

    except Exception as e:
        logger.error(f"Translation failed: {e}")
        raise RuntimeError(f"Translation failed: {e}") from e