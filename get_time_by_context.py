from transformers import BlipProcessor, BlipForConditionalGeneration
import cv2
import torch
import pandas as pd
import translators as ts
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity


ts.translate_text('text', translator='google', to_language='ru')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
photo_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


# Функция для получения эмбеддинга
def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    # Извлечение эмбеддинга: среднее значение по всем токенам
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()  # Удаляем лишние размерности


def get_recs(new_description, descriptions, n_recs=1):
    '''Получение рекомендаций видео для промта'''
    
    new_embedding = get_embedding(new_description)

    #получение эмбендингов для описаний из бд
    embeddings = [get_embedding(desc) for desc in descriptions]

    # Вычисление схожести
    similarities = cosine_similarity(new_embedding.reshape(1, -1), embeddings)
    most_similar_index = similarities.argsort()[0][-n_recs:][::-1]
    
    # most_similar_description = [descriptions[i] for i in most_similar_index]
    
    return most_similar_index



def find_second_by_get_info_on_photo(video_path, promt, transcribitions, n_recomendations=1):
    cap = cv2.VideoCapture(video_path)
    frames = cap.get(cv2.CAP_PROP_FPS)
    data = pd.read_csv(transcribitions)
    cnt = 0
    descriptions_of_cadrs = []
    part_of_trans = 0
    flag = True
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if cnt % frames == 0:
            print(int(cnt // frames))
            part_of_trans_full = list(data.loc[part_of_trans])
            if flag:
                if int(cnt // frames) > part_of_trans_full[1]:
                    part_of_trans+=1
            if part_of_trans == data.shape[0] - 1:
                flag=False
            text = part_of_trans_full[2]

            inputs = processor(frame, return_tensors="pt")

            with torch.no_grad():
                outputs = photo_model.generate(**inputs)
            
            descr = processor.decode(outputs[0], skip_special_tokens=True)
            text_res = text + ts.translate_text(descr, translator='google', to_language='ru')

            descriptions_of_cadrs.append(text_res)
            
        cnt += 1
    
    result = get_recs(promt, descriptions_of_cadrs, n_recomendations)

    return result