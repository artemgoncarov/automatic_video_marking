import cv2
from ultralytics import YOLO
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import translators as ts
from iteract import interact
from getRiski import background_risky


model = YOLO('yolov8x.pt')
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
photo_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def detect_objects(video_path):
    cap = cv2.VideoCapture(video_path)
    detected_objects = dict()
    object_frames = dict()
    cnt = 0

    frames = cap.get(cv2.CAP_PROP_FPS)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if cnt % frames == 0:
            print(cnt)
            results = model(frame)

            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    proba = box.conf[0]

                    if proba > 0.7:
                        if class_name in detected_objects:
                            detected_objects[class_name] += 1
                        else:
                            detected_objects[class_name] = 1
                        
                        if class_name in object_frames:
                            object_frames[class_name].add(cnt)
                        else:
                            object_frames[class_name] = {cnt}

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cnt += 1

    cap.release()
    cv2.destroyAllWindows()

    frame_counts = {key: len(value) for key, value in object_frames.items()}

    objects = dict()
    
    for obj in sorted(detected_objects, key=lambda x: detected_objects[x], reverse=True):
        objects[ts.translate_text(obj, translator='google', to_language='ru')] = {
            'total_count': detected_objects[obj],
            'frame_count': frame_counts.get(obj, 0)
        }
        
    return {"objects": objects, "total_frames": cnt // frames}



def generate_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    cnt = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if cnt % cap.get(cv2.CAP_PROP_FPS) == 0:
            results = model(frame)  # Предполагаем, что модель уже загружена

            # Аннотируем кадр
            annotated_frame = results[0].plot()

            # Конвертируем кадр в JPEG
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()

            # Генерируем поток изображения
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        cnt += 1

    cap.release()


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    detected_objects = dict()
    cnt = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if cnt % cap.get(cv2.CAP_PROP_FPS) == 0:
            results = model(frame)

            # Обработка обнаруженных объектов
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    proba = box.conf[0]

                    if proba > 0.7:
                        if class_name in detected_objects.keys():
                            detected_objects[class_name] += 1
                        else:
                            detected_objects[class_name] = 1

            # Аннотируем кадр
            annotated_frame = results[0].plot()

            # Конвертируем кадр в JPEG
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame = buffer.tobytes()

            # Генерируем поток изображения
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        cnt += 1

    cap.release()
    
    # Формируем итоговые данные о обнаруженных объектах
    objects = dict()
    for i in sorted(detected_objects, key=lambda x: detected_objects[x][1])[::-1]:
        objects[i] = detected_objects[i]

    return objects


def get_info_on_photo(video_path, second):
    cap = cv2.VideoCapture(video_path)
    time = second * cap.get(cv2.CAP_PROP_FPS)

    cnt = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if cnt == time:
            inputs = processor(frame, return_tensors="pt")

            with torch.no_grad():
                outputs = photo_model.generate(**inputs)
            
            descr = processor.decode(outputs[0], skip_special_tokens=True)

            cv2.imwrite('frame.jpeg', frame)

            scene, riski = background_risky('frame.jpeg')

            descr = ts.translate_text(descr, translator='google', to_language='ru')
            scene = ts.translate_text(scene, translator='google', to_language='ru')
            if riski != "nothing risky":
                riski = ts.translate_text(riski, translator='google', to_language='ru')
            else:
                riski = "Рисков нет"

            return (descr, scene, riski)
        
        cnt += 1
