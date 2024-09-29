from flask import Flask, render_template, request, Response, jsonify, send_file
import os
from detecting_objects import generate_frames, detect_objects, get_info_on_photo
import csv
import pandas as pd
from get_text import *
from razmetka_text import *
from report import create_report
from tempGrams import apply_grad_cam_to_folder
from get_text_from_image import extract_text_with_preprocessing
from get_time_by_context import find_second_by_get_info_on_photo
from audio_report import get_audio_report

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    file = request.files['video']
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        return jsonify({'video_url': file_path})
    return jsonify({'error': 'No file uploaded'}), 400


@app.route('/transcribe', methods=['GET'])
def download_transcription():
    path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], 'transcription.csv')
    
    return send_file(path_to_save, as_attachment=True, download_name='transcription.csv')

@app.route('/getInfo', methods=['GET'])
def get_info():
    second = int(float(request.args.get('second')))
    video = request.args.get('video_path')

    print(second, video)

    descr = get_info_on_photo(video, second)
    apply_grad_cam_to_folder('frame.jpeg', 243)
    text = extract_text_with_preprocessing('frame.jpeg')

    return jsonify({'description': descr, 'path': os.path.join(app.config['UPLOAD_FOLDER'], 'image.jpeg'), "text": text}), 200


@app.route('/getSecond', methods=['GET'])
def get_second():
    video = request.args.get('video')
    text = request.args.get('text')

    seconds = list(map(lambda x: str(x), (find_second_by_get_info_on_photo(video, text, 'audio.tsv', 3))))

    print(seconds)

    return jsonify({'second1': seconds[0], 'second2': seconds[1], 'second3': seconds[2]}), 200


@app.route('/report', methods=['GET'])
def download_report():
    path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], 'example.docx')
    
    return send_file(path_to_save, as_attachment=True, download_name='example.docx')

@app.route('/audio', methods=['GET'])
def download_audio():
    return send_file('example.json', as_attachment=True, download_name='example.json')


@app.route('/tempImg', methods=['GET'])
def download_tempimg():
    path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], 'image.jpeg')
    
    return send_file(path_to_save, as_attachment=True, download_name='image.jpeg')

@app.route('/generate_transcription', methods=['POST'])
def generate_transcription():
    video_path = request.json.get('video_url')
    path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], 'transcription.csv')

    create_model()
    get_text(video_path, path_to_save)
    del_model()
    create_report(path_to_save, os.path.join(app.config['UPLOAD_FOLDER'], 'example.docx'))
    get_audio_report("audio.mp3")
    
    return jsonify({'message': 'Transcription generated successfully'}), 200


@app.route('/video_feed')
def video_feed():
    video_path = request.args.get('video_url')
    return Response(generate_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/detect_objects', methods=['POST'])
def detect_objects_route():
    video_path = request.json.get('video_url')
    detected_objects = detect_objects(video_path)
    
    return jsonify({'detected_objects': detected_objects}), 200


if __name__ == '__main__':
    app.run(port=1488, debug=True)