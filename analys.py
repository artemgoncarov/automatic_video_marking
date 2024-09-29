import flask
import os
from flask import Flask, render_template, request, Response, jsonify, send_file
from razmetka_text import *


app = Flask(__name__)

@app.route('/analyze_transcription', methods=['POST'])
def analyze_transcription():
    path_to_text = os.path.join(app.config['UPLOAD_FOLDER'], 'transcription.csv')
    
    # Выполняем анализ текста
    analysis_result = main_analys(path_to_text)
    
    # Возвращаем результат в формате JSON
    return jsonify({'analysis_result': analysis_result}), 200


if __name__ == '__main__':
    app.run(debug=True)