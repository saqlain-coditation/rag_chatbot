import os
import secrets
import string
from pathlib import Path

from google import genai
from PIL.ImageFile import ImageFile

import config

IMAGE_PROMPT_TEMPLATE = """
You are an AI assistant. You have been given an image of a document.
Extract and summarize all meaningful information 
from the given image. Transcribe text accurately. 
If it contains diagrams, tables, or charts, describe them clearly.

Return only the final description and nothing else.
If the image is a graphic and not a document then skip and Return nothing.
Don't add any markdown.
"""

PIL_FORMAT_TO_MIME = {
    "JPEG": "image/jpeg",
    "JPG": "image/jpeg",
    "PNG": "image/png",
    "GIF": "image/gif",
    "BMP": "image/bmp",
    "TIFF": "image/tiff",
    "WEBP": "image/webp",
}


def ocr(image: ImageFile) -> str | None:

    response = config.genai_client.models.generate_content(
        model=config.vision_llm_model,
        contents=[
            genai.types.Part.from_text(text=IMAGE_PROMPT_TEMPLATE),
            image,
        ],
    )

    text = response.text
    filename = image.filename or "".join(
        secrets.choice(string.ascii_letters + string.digits) for _ in range(6)
    )
    img_format = image.format or "JPEG"
    mime = PIL_FORMAT_TO_MIME.get(img_format.upper(), PIL_FORMAT_TO_MIME["JPEG"])
    ext = mime.split("/")[-1]
    filepath = Path(f".logs/images/{filename}.{ext}")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    image.save(filepath, format=image.format)
    return text
