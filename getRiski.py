import translators as ts
from PIL import Image
from transformers import pipeline

def background_risky(image):
    vqa_pipeline = pipeline("visual-question-answering")
    question = "What is the background in this image?"
    scene = (vqa_pipeline(image, question, top_k=1)[0]['answer'])
    question = "Is there anything risky in the image? If so, then write that"
    risky = vqa_pipeline(image, question, top_k=1)[0]['answer']
    print(risky)
    if risky != 'no':
        question = "What is risky?"
        risk = "risky " + vqa_pipeline(image, question, top_k=1)[0]['answer']
    else:
        risk = "nothing risky"

    return (scene, risk)