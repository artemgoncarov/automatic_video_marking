# Разметка видеоконтента

## Решение команды [RASCAR] fsociety
### Состав и роли:
Артём Гончаров - Team Lead, Frontend, NLP, CV.
Андрей Алёхин - CV, NLP.
Игорь Карташов - CV, NLP.
Артём Гуйван - Audio, NLP.


# Проблематика

В рамках проектов Газпром Медиа Холдинга требуется масштабная разметка видео-контента, как
профессионального, так и пользовательского (UGC). Автоматизация процесса разметки является
ключевой задачей для снижения трудозатрат и повышения качества обработки данных. В условиях
растущих объемов видеоматериалов необходимо разработать решение, которое позволит быстро и
эффективно разметить контент, выделяя ключевые объекты, события и сцены, а также проводить
анализ тональности, символов и звукового сопровождения. Основные задачи: повышение качества
разметки, снижение стоимости и увеличение скорости обработки контента.

# Постановка задачи

Задача заключается в разработке решений для автоматизации
разметки большого объема видео-контента, включая
профессиональные и пользовательские видеоматериалы.
Требуется продемонстрировать эффективные методы разметки,
которые позволят значительно улучшить качество, снизить
затраты и ускорить процесс разметки. Решение должно включать
инструменты для транскрибации голоса, анализа звукового
сопровождения, выявления объектов, символов и сцен, а также
организации поиска по видео. Участники должны обладать
собственным программно-аппаратным комплексом для
выполнения задания.

# Описание решения

Представляем вам наше инновационное решение, которое существенно упрощает и ускоряет разметку видеоконтента через удобный веб-интерфейс. В его основе лежат передовые технологии: stable-whisper для транскрибации, saiga llama3 8b llm для выделения ключевых событий по тексту, yolo v8x для детекции объектов и blip для анализа содержимого на изображениях. Уникальность нашего подхода заключается в мультимодальности моделей, которые могут работать с контентом на любом языке, а также в способности предоставлять детализированную информацию о событиях в конкретные моменты времени. Дополнительно наше решение предлагает богатый арсенал методов для сбора статистики, что делает процесс разметки видео быстрым и эффективным. Мы уверены, что оно значительно повысит вашу продуктивность.

# Киллерфичи

### Особенности нашего решения:

- Возможность получения информации о происходящем на видео в определенный момент времени
- Поиск моментов на видео по содержимому контекста
- Мультилингвистические модели
- Детекция объектов при вероятности > 0.7
- Решение использует только опен-сорс технологии
- Решение не требует постоянного подключения к сети
- Все модели можно использовать в бизнесовых задачах

# Масштабируемость

### Планы на развитие проекта в будущем:



# Стек технологий

Мы используем такие технологии, как:
- flask, bootstap(frontent)
- saiga llama3 8b, stable-whisper, resnet50, yolo, blip, transformers(backend, ml)

# Run


## Скачайте модели
```
wget https://huggingface.co/artemgoncarov/cp_vseross_2024/resolve/main/model-q2_K.gguf
wget https://huggingface.co/artemgoncarov/cp_vseross_2024/resolve/main/yolov8x.pt
```

## Virtual environment

Мы предлагаем вам использовать Python версии 3.11.

### Создание виртуального окружения
```
python -m venv venv
```
### Активация окружения

#### Windows:
```
venv\Scripts\activate
```
#### Linux/MacOS:
```
source venv/bin/activate
```

### Установка библиотек

```
pip install -r requirements.txt
```

## Запуск

```
python app.py
```

Далее у вас будет доступен веб-интерфейс по локальному адресу http://127.0.0.1:5000


# Описание файлов

## app.py

Главный файл - запуск веб-интерфейса.

## audio_report.py

Составление отчета по аудио дорожке видео.

## detecting_objects.py

Файл с функциями детекции объектов.

## get_text_from_image.py

Получение информации по фото.

## get_text.py

Транскрибация аудио.

## get_time_by_context.py

Получение информации в определенный промежуток времени.

# getRiski.py

Получение рисков на фото.

## report.py

Файл с функцией для составления отчетов на базе транскрибации.

## tempGrams.py

Создание теплограмм.

## iteract.py

Файл с функцией отправки запроса в LLM.

# Рекомендуем к подписке на ТГ-каналы нашего сообщества

https://t.me/rascar_ai
