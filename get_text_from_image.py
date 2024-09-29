import cv2
import pytesseract
from PIL import Image
from transformers import pipeline
vqa_pipeline = pipeline("visual-question-answering")


def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, img_processed = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    
    return img_processed

def extract_text_with_preprocessing(img):
    question = "Is there the text on image? If so, write it"
    ifText = vqa_pipeline(img, question, top_k=1)[0]['answer']
    if ifText == 'no':
        return ''
    img = preprocess_image(img)
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(img, config=custom_config, lang='rus')

    print(text)
    
    return text