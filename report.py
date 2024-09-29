import string
import pandas as pd
from razmetka_text import *
import pymorphy3
from nltk.corpus import stopwords
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from docx import Document


nltk.download('stopwords')
stop_words = set(stopwords.words('russian'))
morph = pymorphy3.MorphAnalyzer()


def save_to_word(text, file_name):
    doc = Document()
    doc.add_paragraph(text)
    doc.save(file_name)


def preprocess_text(text):
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    
    filtered_words = [word for word in words if word.lower() not in stop_words]
    
    pos_filtered_words = []
    for word in filtered_words:
        parsed_word = morph.parse(word)[0]
        if parsed_word.tag.POS in {'NOUN', 'ADJF', 'VERB', 'INFN', 'PRTF', 'PRTS'}:
            pos_filtered_words.append(parsed_word.normal_form)
    
    return ' '.join(pos_filtered_words)


def get_words(text):
    preprocessed_text = preprocess_text(text)
    vectorizer = TfidfVectorizer(max_features=10)
    tfidf_matrix = vectorizer.fit_transform([preprocessed_text])
    feature_names = vectorizer.get_feature_names_out()
    scores = zip(feature_names, tfidf_matrix[0].toarray().flatten())
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    words = ''
    for word, score in sorted_scores:
        # print(f"{word}: {score:.3f}")
        words += f"{word}: {score:.3f}\n"

    return words


def create_report(path_to_text, path_to_save):
    df = pd.read_csv(path_to_text)

    text = ' '.join(list(df['text']))

    a = main_analys(path_to_text)
    b = razmetka(path_to_text)
    
    words = get_words(text)

    text1 = "Полный текст:\n\n" + text + "\n\nКлючевые события:\n\n" + a + '\n\nРазметка:\n\n' + b + "\n\nКлючевые слова:\n\n" + words

    save_to_word(text1, path_to_save)