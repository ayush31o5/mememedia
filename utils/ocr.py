import pytesseract
from PIL import Image
import google.generativeai as genai
import os
import re

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

EMOJI_REGEX = re.compile(
    "[\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002700-\U000027BF"
    "\U000024C2-\U0001F251" 
    "]+", flags=re.UNICODE
)

def contains_emoji(text):
    return bool(EMOJI_REGEX.search(text))

def extract_text_from_image_tesseract(image_path, language='eng'):
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image, lang=language)
        return text.strip()
    except Exception as e:
        print(f"Error during Tesseract OCR: {e}")
        return None

def extract_text_from_image_gemini(image_path, api_key=None, model_name="gemini-1.5-flash", prompt="Extract the text from this image, including any emojis."):
    if api_key is None:
        api_key = 'AIzaSyAHBYZGkBWwBaSCt4rXyvDA3sQfjSwJGro'
        if not api_key:
            raise ValueError("Google Gemini API key not provided and GOOGLE_API_KEY not set.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    try:
        img = Image.open(image_path)
        response = model.generate_content([prompt, img])
        response.resolve()
        return response.text.strip()
    except Exception as e:
        print(f"Error during Gemini OCR: {e}")
        return None

def ocr_multi_lingual(image_path, tesseract_language='eng', gemini_api_key=None, gemini_prompt=None, use_tesseract_first=True):
    text = None

    if use_tesseract_first:
        text = extract_text_from_image_tesseract(image_path, tesseract_language)
        if text:
            print("Tesseract result:", text)
            if len(text) < 10 or contains_emoji(text) or "unreadable" in text.lower():
                print("Detected emoji or poor Tesseract result. Trying Gemini...")
                text = None
            else:
                return text

    print("Falling back to Google Gemini OCR...")
    text = extract_text_from_image_gemini(
        image_path,
        api_key=gemini_api_key,
        prompt=gemini_prompt or "Extract the text from this image, including any emojis."
    )
    if text:
        print("Gemini result:", text)
        return text
    else:
        print("Gemini also failed to extract text.")
        return None