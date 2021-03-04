import pyaudio
import wave
from model import Res15
from manage_audio import AudioPreprocessor
import librosa
import torch
import time
import numpy as np
import os
import soundfile as snd
from py_webrtcvad import vad
import shutil

class Recorder():
    def __init__(self, record_seconds=3, channel=2, rate=44100, chunk=1024, format=pyaudio.paInt16):
        self.chunk = chunk
        self.format = format
        self.channel = channel
        self.rate = rate
        self.record_seconds = record_seconds

    def record(self, output_file):
        precessor = pyaudio.PyAudio()
        stream = precessor.open(format=self.format,
                                     channels=self.channel,
                                     rate=self.rate,
                                     input=True,
                                     frames_per_buffer=self.chunk)
        print("After 3 seconds, recording begins, are you ready?")
        time.sleep(1)
        print("3...")
        time.sleep(1)
        print("2...")
        time.sleep(1)
        print("1...")
        time.sleep(1)
        print("-------------------recording-------------------")
        frames = []

        for i in range(0, int(self.rate / self.chunk * self.record_seconds)):
            data = stream.read(self.chunk)
            frames.append(data)

        print("----------------done recording----------------")

        stream.stop_stream()
        stream.close()
        precessor.terminate()

        wf = wave.open(output_file, 'wb')
        wf.setnchannels(self.channel)
        wf.setsampwidth(precessor.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(frames))
        wf.close()

class EnrollmentProcessor():
    def __init__(self, model_file="trained_models/res15_fine_grained.pt", record_length=3, channel=1, sr=16000):
        self.model = Res15(n_labels=26)
        self.model.load(model_file)
        self.audio_processor = AudioPreprocessor()
        self.sr = sr
        self.recorder = Recorder(record_seconds=record_length, channel=channel, rate=sr)
        os.makedirs("temp", exist_ok=True)

    def data_augment(self, recording_dir):
        # clear old files and copy original enrollment audio
        if os.path.exists("augmented"):
            shutil.rmtree("augmented")
        os.mkdir("augmented")
        for file in os.listdir(recording_dir):
            shutil.copy(os.path.join(recording_dir, file), os.path.join("augmented", file))
        # data augment
        for file in os.listdir("augmented"):
            input_file = os.path.join("augmented", file)
            os.system("sox " + input_file + " " + input_file.replace(".wav", "_gain3dB.wav") + " gain 3")
            os.system("sox " + input_file + " " + input_file.replace(".wav", "_gain-3dB.wav") + " gain -3")
            os.system("sox " + input_file + " " + input_file.replace(".wav", "_tempo125.wav") + " tempo 1.25")
            os.system("sox " + input_file + " " + input_file.replace(".wav", "_tempo80.wav") + " tempo 0.8")

    def compute_embedding(self, file_name, in_len=16000):
        data, sr = librosa.load(file_name, sr=self.sr)
        if len(data) <= in_len:
            data = np.pad(data, (0, in_len - len(data)), "constant")  # data padding
        else:
            data = data[:in_len]  # data truncating
        x = self.audio_processor.compute_mfccs(data).reshape(1, 101, 40)
        os.makedirs("features/train/personalized_keyword", exist_ok=True)
        np.save(os.path.join("features/train/personalized_keyword", os.path.basename(file_name).replace(".wav", ".npy")), x)
        self.model.eval()
        y, embedding = self.model(torch.from_numpy(x))
        return embedding

    def enroll_speech(self, n=3, record=True, data_augment=False, vad_process=False):
        buffer = []
        if record:
            data_augment = True
            vad_process = True
            for i in range(n):
                self.recorder.record("temp/enroll_{}.wav".format(i))
        if data_augment:
            print("--------data augment----------")
            self.data_augment("temp")
            assert len(os.listdir("augmented")) == n * 5  # check whether the samples are augmented successfully
        # vad processing
        if vad_process:
            print("--------vad process----------")
            if os.path.exists("trimmed"):   # clear old files
                shutil.rmtree("trimmed")
            os.makedirs("trimmed", exist_ok=True)
            for file in os.listdir("augmented"):
                file_name = os.path.join("augmented", file)
                vad(file_name, os.path.join("trimmed", os.path.basename(file_name)), aggressive=3)
            assert len(os.listdir("trimmed")) == n * 5  # check whether there are empty utterance
        # computing embedding
        for file in os.listdir("trimmed"):
            embedding = self.compute_embedding(os.path.join("trimmed", file))
            buffer.append(embedding)
        avg_embedding = torch.mean(torch.cat(buffer, dim=0), dim=0).unsqueeze(0)
        torch.save(avg_embedding, "personalized_keyword.pt")
        print("Enrollment finished.")
        return avg_embedding

if __name__ == '__main__':
    p = EnrollmentProcessor()
    p.enroll_speech(n=3, record=True)

    # import torch.nn.functional as F
    # embedding = p.compute_embedding("go.wav")
    # y = torch.load("personalized_keyword.pt")
    # score = F.cosine_similarity(embedding, y)
    # print(score)
    pass