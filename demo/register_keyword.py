import pyaudio
import wave
from model import Res15
from manage_audio import AudioPreprocessor
import librosa
import torch
import time
import numpy as np

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
    def __init__(self, model_file="res15_fine_grained.pt", record_length=3, channel=1, sr=16000):
        self.model = Res15(n_labels=26)
        self.model.load(model_file)
        self.audio_processor = AudioPreprocessor()
        self.sr = sr
        self.recorder = Recorder(record_seconds=record_length, channel=channel, rate=sr)

    def compute_embedding(self, file_name, in_len=16000):
        data, sr = librosa.load(file_name, sr=self.sr)
        data, _ = librosa.effects.trim(data)
        if len(data) <= in_len:
            data = np.pad(data, (0, in_len - len(data)), "constant")  # data padding
        else:
            data = data[:in_len]  # data truncating
        x = self.audio_processor.compute_mfccs(data).reshape(1, 101, 40)
        self.model.eval()
        y, embedding = self.model(torch.from_numpy(x))
        return embedding

    def enroll_speech(self, n=3, keyword="none", record=True):
        buffer = []
        for i in range(n):
            if record:
                self.recorder.record("temp_{}.wav".format(i))
            embedding = self.compute_embedding("temp_{}.wav".format(i))
            buffer.append(embedding)
        avg_embedding = torch.mean(torch.cat(buffer, dim=0), dim=0).unsqueeze(0)
        torch.save(avg_embedding, keyword + ".pt")
        print("Enrollment finished.")
        return avg_embedding

if __name__ == '__main__':
    import torch.nn.functional as F

    p = EnrollmentProcessor()
    # p.enroll_speech(n=3, keyword="open", record=False)
    embedding = p.compute_embedding("go.wav")
    y = torch.load("open.pt")
    score = F.cosine_similarity(embedding, y)
    print(score)
    pass