import os
import librosa
import numpy as np
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow_io as tfio
import glob
from typing import List
from dataclasses import dataclass
import IPython.display as ipd
from speechbrain.inference.interfaces import foreign_class
import soundfile as sf
import numpy as np
import os
import glob
import shutil
import json


class AudioClassifierYAMNet:
    def __init__(self, model_url):
        self.model = hub.load(model_url)
        class_map_path = self.model.class_map_path().numpy().decode('utf-8')
        self.class_names = list(pd.read_csv(class_map_path)['display_name'])

    @tf.function
    def load_wav_16k_mono(self, filename, sample_rate=16000):
        """
        Загружает WAV файл, преобразует его в одномерный тензор с частотой дискретизации 16 kHz.

        :param filename: Путь к файлу WAV.
        :param sample_rate: Частота дискретизации.
        :return: аудио в формате Tensor.
        """
        file_contents = tf.io.read_file(filename)
        wav, original_sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
        wav = tf.squeeze(wav, axis=-1)
        original_sample_rate = tf.cast(original_sample_rate, dtype=tf.int64)
        wav = tfio.audio.resample(wav, rate_in=original_sample_rate, rate_out=sample_rate)
        return wav

    def classify(self, wav_data):
        """
        Классифицирует аудиоданные и возвращает основной класс.

        :param wav_data: Аудиоданные в формате Tensor.
        :return: Предсказанный класс звука.
        """
        scores, embeddings, spectrogram = self.model(wav_data)
        class_scores = tf.reduce_mean(scores, axis=0)
        top_class = tf.argmax(class_scores)
        inferred_class = self.class_names[top_class]

        return inferred_class, embeddings

    def plot_audio(self, wav_data):
        """
        Строит график аудиоданных.

        :param wav_data: Аудиоданные в формате Tensor.
        """
        plt.plot(wav_data)
        plt.show()

    def play_audio(self, wav_data, sample_rate=16000):
        """
        Проигрывает аудиоданные.

        :param wav_data: Аудиоданные в формате Tensor.
        :param sample_rate: Частота дискретизации.
        """
        return ipd.Audio(wav_data, rate=sample_rate)

@dataclass
class Config:
    speech_classes: List[str] = (
        'Speech', 'Child speech, kid speaking', 'Conversation', 'Narration, monologue',
        'Babbling', 'Speech synthesizer', 'Shout', 'Bellow', 'Whoop', 'Yell', 
        'Children shouting', 'Screaming', 'Whispering', 'Laughter', 'Baby laughter', 
        'Giggle', 'Snicker', 'Belly laugh', 'Chuckle, chortle', 'Crying, sobbing', 
        'Baby cry, infant cry', 'Whimper', 'Wail, moan', 'Sigh', 'Singing', 'Choir', 
        'Yodeling', 'Chant', 'Mantra', 'Child singing', 'Synthetic singing', 'Rapping', 
        'Humming', 'Groan', 'Grunt', 'Whistling', 'Breathing', 'Wheeze', 'Snoring', 
        'Gasp', 'Pant', 'Snort', 'Cough', 'Throat clearing', 'Sneeze', 'Sniff', 
        'Stomach rumble', 'Burping, eructation', 'Hiccup', 'Clapping', 'Cheering', 
        'Applause', 'Chatter', 'Crowd', 'Hubbub, speech noise, speech babble', 
        'Children playing', 'Music'
    )

class SpeechClassifier:
    def __init__(self):
        self.config = Config()
        # Замена на правильный источник модели
        self.classifier = foreign_class(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", 
                                        pymodule_file="custom_interface.py", 
                                        classname="CustomEncoderWav2vec2Classifier")

    def classify(self, filepath):
        out_prob, score, index, text_lab = self.classifier.classify_file(filepath)
        if text_lab in self.config.speech_classes:
            return text_lab
        return "no speech"


class Pipeline:
    def __init__(self, file_name, yamnet_model_handle='https://tfhub.dev/google/yamnet/1'):
        self.file_name = file_name
        self.signal, self.sr = librosa.load(file_name)  # Загрузка аудио
        self.yamnet_model_handle = yamnet_model_handle
        self.audio_classifier = AudioClassifierYAMNet(yamnet_model_handle)
        self.report = {"sentiment_analysis": None,
                       "count_key_peaks": None,
                       "list_key_peaks": None,
                       "classification": None}
        self.padded_timestampes = []

    def filter_close_peaks(self, min_distance=100):
        """
        Фильтрует индексы пиков, чтобы сохранялись только те, которые находятся на расстоянии не менее min_distance.
        """
        cent = librosa.feature.spectral_centroid(y=self.signal, sr=self.sr)
        cent_diff = np.diff(cent)
        threshold = np.std(cent_diff) * 3
        peaks = np.where(np.abs(cent_diff) > threshold)[1]
        filtered_peaks = [peaks[0]]
        for peak in peaks[1:]:
            if peak - filtered_peaks[-1] >= min_distance:
                filtered_peaks.append(peak)
        return filtered_peaks

    def get_peaks_in_seconds(self, filtered_peaks):
        """
        Конвертирует индексы пиков в секунды.
        """
        hop_length = 512
        peaks_in_seconds = [(peak * hop_length) / self.sr for peak in filtered_peaks]
        return peaks_in_seconds

    def get_waveform_timestamp(self, timestamp=50, delta=4, show=False):
        """
        Возвращает сегмент аудиосигнала, соответствующий указанному времени.
        """
        waveform = self.signal
        if timestamp > waveform.shape[0] / self.sr:
            timestamp = waveform.shape[0] / self.sr

        if len(waveform.shape) == 1:
            waveform = np.expand_dims(waveform, 0)

        num_channels, num_frames = waveform.shape
        time_axis = np.arange(0, num_frames) / self.sr

        bound_left = int(max(0, timestamp * self.sr - delta * self.sr))
        bound_right = int(min(num_frames, timestamp * self.sr + delta * self.sr))

        if show:
            plt.plot(time_axis[bound_left:bound_right], waveform[0][bound_left:bound_right])
            plt.show()

        return waveform[:, bound_left:bound_right]

    def pad_segments(self, timestampes):
        """
        Выравнивает длины всех сегментов путем добавления нулевых значений.
        """
        max_len = max([segment.shape[1] for segment in timestampes])

        padded_timestampes = []
        for segment in timestampes:
            padding = np.zeros((segment.shape[0], max_len - segment.shape[1]))
            padded_segment = np.concatenate((segment, padding), axis=1)
            padded_timestampes.append(padded_segment)

        self.padded_timestampes = np.array(padded_timestampes)

    def process(self):
        """
        Основной процесс анализа аудио, включая фильтрацию пиков, выделение сегментов и их классификацию.
        """
        filtered_peaks = self.filter_close_peaks()
        peaks_in_seconds = self.get_peaks_in_seconds(filtered_peaks)

        self.report["list_key_peaks"] = [round(peak, 2) for peak in peaks_in_seconds]
        self.report["count_key_peaks"] = len(peaks_in_seconds)

        timestampes = []
        for peak in peaks_in_seconds:
            segment = self.get_waveform_timestamp(timestamp=peak, show=False)
            timestampes.append(segment)

        self.pad_segments(timestampes)

        # Классификация сегментов
        self.report["classes"] = [0 for _ in range(len(self.report["list_key_peaks"]))]
        for number, pad_timestampe in enumerate(self.padded_timestampes):
            pad_timestampe = np.squeeze(pad_timestampe)
            inferred_class, _ = self.audio_classifier.classify(pad_timestampe)
            self.report["classes"][number] = inferred_class

        return self.report

def get_audio_report(audio_path):
    speech_classifier = SpeechClassifier()

    padded_timestampes = np.random.randn(14, 1, 176400)
    output_dir = "audio_files"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(padded_timestampes.shape[0]):
        audio_data = padded_timestampes[i, 0, :]
        file_name = os.path.join(output_dir, f"audio_{i}.wav")
        sf.write(file_name, audio_data, 16000)
        print(f"Created file: {file_name}")

    pipeline = Pipeline(audio_path)
    report = pipeline.process()

    report["sentiment_analysis"] = [0 for i in range(len(report["list_key_peaks"]))]

    for num, i in enumerate(glob.glob("/content/audio_files/*")):
        report["sentiment_analysis"][num] = speech_classifier.classify(i)

    with open("example.json", "w", encoding="utf-8") as file:
        json.dump(report, file)