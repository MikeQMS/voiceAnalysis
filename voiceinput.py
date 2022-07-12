import time
import pyaudio
from predicting import predicting
import numpy as np
import struct
import matplotlib.pyplot as plt


class AudioStreamBuilder:
    def __init__(self, chunk, format, channels, rate, record_seconds, input, output):
        self.p = pyaudio.PyAudio()
        self.chunk = chunk
        self.format = format
        self.channels = channels
        self.rate = rate
        self.record_seconds = record_seconds
        self.input = input
        self.output = output

    def open(self):
        return self.p.open(format=self.format, channels=self.channels, rate=self.rate, input=self.input,
                           output=self.output, frames_per_buffer=self.chunk)

    def terminate(self):
        self.p.terminate()


CHUNK = 2048
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 60

analysis = AudioStreamBuilder(CHUNK, pyaudio.paFloat32, CHANNELS, RATE, RECORD_SECONDS, True, True)
display = AudioStreamBuilder(CHUNK, pyaudio.paInt16, CHANNELS, RATE, RECORD_SECONDS, True, False)

datastream_analysis = analysis.open()
datastream_display = display.open()

print("* recording")

frames = []
start = time.time()
new_startframe = 0


fig, ax = plt.subplots()
x = np.arange(0, 2*CHUNK, 2)
line, = ax.plot(x, np.random.rand(CHUNK), 'r')
ax.set_ylim(-60000, 60000)
ax.ser_xlim = (0, CHUNK)
fig.show()

prediction = [""]

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = datastream_analysis.read(CHUNK)
    frames.append(data)
    if time.time() - start >= 2:
        print("start" + str(new_startframe) + "end " + str(len(frames)))
        prediction = predicting(data, RATE)
        new_startframe = len(frames)
        start = time.time()
    data2 = datastream_display.read(CHUNK)
    # datastream_analysis.write(data, CHUNK) # Output voice in headset
    dataInt = struct.unpack(str(CHUNK) + 'h', data2)
    line.set_ydata(dataInt)
    plt.title(prediction[0])
    fig.canvas.draw()
    fig.canvas.flush_events()

print("* done recording")

datastream_analysis.stop_stream()
datastream_display.stop_stream()
datastream_analysis.close()
datastream_display.close()
analysis.terminate()
display.terminate()




