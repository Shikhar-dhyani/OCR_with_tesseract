from flask import Flask, jsonify
import cv2
import numpy as np
import pytesseract
from PIL import Image
import requests
from io import BytesIO

app = Flask(__name__)

def process_image(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    smooth_image = cv2.medianBlur(binary_image, 1)

    config = '--psm 6'
    pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Shikhar\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
    text = pytesseract.image_to_string(Image.fromarray(smooth_image), lang='eng', config=config)

    return text

@app.route('/extract_text/<path:image_url>')
def extract_text(image_url):
    try:
        response = requests.get(image_url)
        response.raise_for_status() 
        
        image = Image.open(BytesIO(response.content))
        extracted_text = process_image(image)

        return jsonify({'text': extracted_text})
    
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)
