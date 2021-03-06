import tkinter as tk
import threading
from model import Res15
import torch
from manage_audio import AudioPreprocessor
import collections
import time
import pyaudio
import numpy as np
import torch.nn.functional as F

import matplotlib
# matplotlib.use('Agg')  # 该模式下绘图无法显示，plt.show()也无法作用
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)

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
        self.model = Res15(n_labels=26)
        # self.model.load("trained_models/res15_fine_grained.pt")
        # self.labels = ["silence", "_unknown_", "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
        self.model.load("trained_models/res15_fine_grained_fine_tuned.pt")
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
        self.label = "---"
        self.is_customized = False
        self.user_defined_keyword_embedding = torch.load("personalized_keyword.pt")
        self.audio_data = np.zeros(16000)
        self.threshold = 0.7

    def _start(self, sleep_time=0.1):
        self._running = True

        def audio_callback(in_data, frame_count, time_info, status):
            self.ring_buffer.extend(in_data)
            play_data = chr(0) * len(in_data)
            return play_data, pyaudio.paContinue

        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            input=True,
            output=False,
            channels=1,
            format=pyaudio.paInt16,
            rate=16000,
            frames_per_buffer=1024,
            stream_callback=audio_callback)

        self.stream.start_stream()
        print("detecting...")

        while self._running is True:

            raw_data = self.ring_buffer.get()
            self.audio_data = buf_to_float(raw_data)

            if len(self.audio_data) == 16000:
                prob, embedding, _ = self.detector.calculate_probability(self.audio_data)
                self.prob_buffer.extend(prob.detach().numpy())
                prob_window = np.vstack(self.prob_buffer)
                average_prob = np.mean(prob_window, axis=0)
                max_index = np.argmax(average_prob, axis=-1)
                label = self.detector.labels[int(max_index)]
                # 控制是否启用自定义唤醒词
                if self.is_customized:
                    probability = F.cosine_similarity(self.user_defined_keyword_embedding, embedding).item()
                    label = "active"
                else:
                    probability = average_prob[int(max_index)]

                if probability > self.threshold:
                    self.label = label
                else:
                    self.label = 'silence'
                # print(prob)

            if len(self.audio_data) == 0:
                time.sleep(sleep_time)
                continue

    def _quit(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

def thread_it(func, *args):
    t = threading.Thread(target=func, args=args)
    t.setDaemon(True)  # 守护进程，使得主线程结束时，子线程也被杀死
    t.start()

def _quit():
    window.quit()     # stops mainloop
    window.destroy()  # this is necessary on Windows to prevent Fatal Python Error: PyEval_RestoreThread: NULL tstate

def result_update():
    var.set(processor.label)
    window.after(100, func=result_update)


if __name__ == "__main__":
    processor = HotwordDetector()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # GUI
    window = tk.Tk()
    window.title('keyword spotting')
    window.geometry('300x200')

    var = tk.StringVar()
    text = tk.Label(window, textvariable=var, font=('Consolas', 48))
    text.pack(expand="yes")

    quit_button = tk.Button(window, text="quit", command=_quit)
    quit_button.pack()

    # canvas = FigureCanvasTkAgg(fig, master=window)  # A tk.DrawingArea.
    # canvas.draw()
    # canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    thread_it(func=processor._start)


    window.after(100, func=result_update)
    window.mainloop()


