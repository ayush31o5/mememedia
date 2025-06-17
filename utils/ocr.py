import pytesseract
from PIL import Image
import google.generativeai as genai
import os

# Ensure the correct Tesseract path is set
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text_from_image_tesseract(image_path, language='eng'):
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image, lang=language)
        return text.strip()
    except Exception as e:
        print(f"Error during Tesseract OCR: {e}")
        return None

def extract_text_from_image_gemini(image_path, api_key=None, model_name="gemini-1.5-flash", prompt="What is written in this image?"):
    if api_key is None:
        api_key = os.getenv('GOOGLE_API_KEY', 'default_api_key')
        if api_key is None:
            raise ValueError("Google Gemini API key not provided and GOOGLE_API_KEY environment variable not set.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    try:
        img = Image.open(image_path)
        response = model.generate_content([prompt, img])
        response.resolve()
        return response.text.strip()
    except Exception as e:
        print(f"Error during Gemini Pro Vision OCR: {e}")
        return None

def ocr_multi_lingual(image_path, tesseract_language='eng', gemini_api_key=None, gemini_prompt="What is written in this image?", use_tesseract_first=True):
    text = None
    if use_tesseract_first:
        text = extract_text_from_image_tesseract(image_path, tesseract_language)
        if text:
            print("Tesseract result:", text)
            if len(text) < 10 or "unreadable" in text.lower():
                print("Tesseract's result is poor, trying Gemini.")
                text = None
            else:
                return text
    if text is None:
        print("Falling back to Google Gemini Pro Vision OCR...")
        text = extract_text_from_image_gemini(image_path, api_key=gemini_api_key, prompt=gemini_prompt)
        if text:
            print("Gemini result:", text)
            return text
        else:
            print("Gemini also failed to extract text.")
            return None
    return text

if __name__ == '__main__':
    image_path = "test.png"
    if not os.path.exists(image_path):
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new('RGB', (400, 100), color = 'white')
        d = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 30)
        except OSError:
            font = ImageFont.load_default()
        d.text((10,10), "Hello World en Español", fill='black', font=font)
        img.save(image_path)
    api_key = os.getenv('GOOGLE_API_KEY', 'your_api_key_here')  # Use environment variable
    if api_key:
        extracted_text = ocr_multi_lingual(image_path, tesseract_language='eng+spa', gemini_api_key=api_key, gemini_prompt="Extract the text from this image.  Return only the text.", use_tesseract_first=True)
        if extracted_text:
            print("Extracted Text:\n", extracted_text)
        else:
            print("OCR failed to extract text.")
    else:
        print("Please set the GOOGLE_API_KEY environment variable or provide the API key directly.")
