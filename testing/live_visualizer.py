import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pylsl import StreamInlet, resolve_streams

# Resolve the stream
streams = resolve_streams()
inlet = StreamInlet(streams[0])

# Setup Plot
fig, ax = plt.subplots()
x_data, y_data = [], []
line, = ax.plot([], [], lw=2)
ax.set_ylim(-100, 100) # Adjust based on your signal range
ax.set_xlim(0, 100)

def update(frame):
    sample, _ = inlet.pull_sample()
    y_data.append(sample[0]) # Assuming channel 0
    if len(y_data) > 100: y_data.pop(0)
    line.set_data(range(len(y_data)), y_data)
    return line,

ani = FuncAnimation(fig, update, interval=10)
plt.show()