
import stable_whisper
import pandas as pd
from moviepy.editor import VideoFileClip
import csv
import torch


# model = stable_whisper.load_model('large')

def extract_audio(video_file, output_audio_file):
    video_clip = VideoFileClip(video_file)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(output_audio_file)
    audio_clip.close()


def get_text(video_path, path_to_save):
    extract_audio(video_path, 'audio.mp3')

    result = model.transcribe("audio.mp3")
    result.to_tsv("audio.tsv")

    start, end, text = [], [], []
    tsv_file = open("audio.tsv", 'r+', encoding='utf-8')
    read_tsv = csv.reader(tsv_file, delimiter="\t")
    for row in read_tsv:
        if len(row) > 0:
            start.append(int(row[0]) / 1000)
            end.append(int(row[1]) / 1000)
            text.append(row[-1])

    d = {'start': start, 'end': end, 'text': text}
    df = pd.DataFrame(data=d)

    df = df[~df['text'].str.isupper()]

    df.to_csv(path_to_save, index=False)

def del_model():
    global model
    del model
    torch.cuda.empty_cache()

def create_model():
    global model
    model = stable_whisper.load_model('large')