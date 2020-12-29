import tkinter as tk
import threading
from model import SpeechResModel, Res15
import torch
from manage_audio import AudioPreprocessor
import collections
import time
import pyaudio
import numpy as np
import torch.nn.functional as F

def buf_to_float(x, n_bytes=2, dtype=np.float32):
    # Invert the scale of the data
    scale = 1. / float(1 << ((8 * n_bytes) - 1))
    # Construct the format string
    fmt = '<i{:d}'.format(n_bytes)
    # Rescale and format the data buffer
    return scale * np.frombuffer(x, fmt).astype(dtype)

class Detector():
    def __init__(self):
        self.audio_processor = AudioPreprocessor()
        # self.config = dict(n_labels=26, use_dilation=True, n_layers=45, n_feature_maps=19)
        # self.model = SpeechResModel(self.config)
        # self.labels = ["silence", "_unknown_", "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]

        self.model = Res15(n_labels=26)
        self.model.load("./res15_fine_grained.pt")
        self.labels = ['forward', 'follow', 'wow', 'marvin', 'learn', 'visual', 'stop', 'house', 'go', 'sheila',
                       'silence', 'cat', 'happy', 'backward', 'yes', 'off', 'bird', 'right', 'left', 'up', 'dog',
                       'bed', 'down', 'on', 'no', 'tree']  # fine-grained label

    def calculate_probability(self, audio_data):
        x = self.audio_processor.compute_mfccs(audio_data).reshape(1, 101, 40)
        self.model.eval()
        y, embedding = self.model(torch.from_numpy(x))
        index = torch.argmax(y)
        label = self.labels[int(index)]
        return y, embedding, label

class RingBuffer():
    """Ring buffer to hold audio from PortAudio"""

    def __init__(self, size=4096):
        self._buf = collections.deque(maxlen=size)

    def extend(self, data):
        """Adds data to the end of buffer"""
        self._buf.extend(data)

    def get(self):
        """Retrieves data from the beginning of buffer and clears it"""
        tmp = bytes(bytearray(self._buf))
        # self._buf.clear()
        return tmp

class HotwordDetector():
    def __init__(self):
        self.ring_buffer = RingBuffer(32000)
        self.prob_buffer = collections.deque(maxlen=3)
        self.detector = Detector()
        self.label = "silence"
        self.user_defined_keyword_embedding = torch.load("open.pt")

    def start_(self, sleep_time=0.1):

        self._running = True

        def audio_callback(in_data, frame_count, time_info, status):
            self.ring_buffer.extend(in_data)
            play_data = chr(0) * len(in_data)
            return play_data, pyaudio.paContinue

        self.audio = pyaudio.PyAudio()
        self.stream_in = self.audio.open(
            input=True,
            output=False,
            channels=1,
            format=pyaudio.paInt16,
            rate=16000,
            frames_per_buffer=1024,
            stream_callback=audio_callback)

        print("detecting...")
        while self._running is True:

            data = self.ring_buffer.get()
            data = buf_to_float(data)

            if len(data) == 16000:
                _, embedding, label = self.detector.calculate_probability(data)
                score = F.cosine_similarity(embedding, self.user_defined_keyword_embedding)
                print(score.item())
                if score.item() < 0.6 and score.item() > 0.5:
                    self.label = "open"
                else:
                    self.label = 'silence'

            if len(data) == 0:
                time.sleep(sleep_time)
                continue

    def start(self, sleep_time=0.1):

        self._running = True

        def audio_callback(in_data, frame_count, time_info, status):
            self.ring_buffer.extend(in_data)
            play_data = chr(0) * len(in_data)
            return play_data, pyaudio.paContinue

        self.audio = pyaudio.PyAudio()
        self.stream_in = self.audio.open(
            input=True,
            output=False,
            channels=1,
            format=pyaudio.paInt16,
            rate=16000,
            frames_per_buffer=1024,
            stream_callback=audio_callback)

        print("detecting...")
        while self._running is True:

            data = self.ring_buffer.get()
            data = buf_to_float(data)

            if len(data) == 16000:
                prob, _, _ = self.detector.calculate_probability(data)
                self.prob_buffer.extend(prob.detach().numpy())
                prob_window = np.vstack(self.prob_buffer)
                average_prob = np.mean(prob_window, axis=0)
                max_index = np.argmax(average_prob, axis=-1)
                label = self.detector.labels[int(max_index)]
                probability = average_prob[int(max_index)]

                if probability > 0.7:
                    print(prob)
                    self.label = label
                else:
                    print(prob)
                    self.label = 'silence'

            if len(data) == 0:
                time.sleep(sleep_time)
                continue

def thread_it(func, *args):
    t = threading.Thread(target=func, args=args)
    t.setDaemon(True)
    t.start()

def result_update():
    var.set(processor.label)
    window.after(100, func=result_update)

if __name__ == "__main__":
    processor = HotwordDetector()
    window = tk.Tk()
    window.title('keyword spotting')
    window.geometry('300x200')
    var = tk.StringVar()
    l = tk.Label(window, textvariable=var, font=('Consolas', 48))
    l.pack(expand='yes')

    thread_it(func=processor.start_)
    window.after(100, func=result_update)
    window.mainloop()

