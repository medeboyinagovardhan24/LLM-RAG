import os
from PIL import Image
import pytesseract
from google import genai
from dotenv import load_dotenv

load_dotenv()

GENAI_KEY = os.getenv("GEMINI_API_KEY")
if not GENAI_KEY:
    raise ValueError("GEMINI_API_KEY not set. Add it to .env or set environment variable.")

client = genai.Client(api_key=GENAI_KEY)
vision_model_name = "gemini-3.6-flash"

def process_image(image_path):
    img = Image.open(image_path)
    text_ocr = pytesseract.image_to_string(img)
    vision_result = client.models.generate_content(
        model=vision_model_name,
        contents=[img]
    )
    text_vision = vision_result.text
    return {"ocr": text_ocr, "vision": text_vision, "combined": text_vision or text_ocr}

def process_scanned_pdf(pdf_path):
    pass
