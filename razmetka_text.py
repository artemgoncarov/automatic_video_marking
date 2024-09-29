import pandas as pd
from iteract import interact

def main_analys(path_to_text):
    df = pd.read_csv(path_to_text)

    text = ' '.join(list(df['text']))  # Убедитесь, что столбец называется 'text'

    res = interact(messages=[
        {
            "role": "system",
            "content": "Ты - ИИ для выявления ключевых событий из видео по его транскрибации. Пользователь тебе предоставляет текст, а ты по нему должен выделить ключевые события, а также указать его тональность - позитив/негатив, уровень тональности."
        },
        {
            "role": "user",
            "content": text
        }
    ])

    return res


def razmetka(path_to_text):
    df = pd.read_csv(path_to_text)

    text = ' '.join(list(df['text']))  # Убедитесь, что столбец называется 'text'

    res = interact(messages=[
        {
            "role": "system",
            "content": "Ты - ИИ для разметки текста. Пользователь тебе предоставляет текст, а тебе нужно выделить в тексте ключевые моменты, основные моменты, 18+, нейтральный контент, не желательный контент."
        },
        {
            "role": "user",
            "content": text
        }
    ])

    return res

