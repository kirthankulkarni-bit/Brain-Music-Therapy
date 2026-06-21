import collections
import threading
import time
import mne
import numpy as np
from pylsl import StreamInlet, resolve_byprop
import os
import csv
from datetime import datetime
import pygame
import sys

# UI stuff
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

# fire up the audio mixer first so we don't get silent failures later
pygame.mixer.init()

# global state tracking
current_state = "ambient"  
system_running = True

# muse 2 hardware specs - do not touch these or the math breaks
sfreq = 128  
seconds_per_chunk = 5
target_buffer_points = sfreq * seconds_per_chunk  

# where the music lives
AUDIO_DIR = "generated_loops"
AMBIENT_PATH = os.path.join(AUDIO_DIR, "calm_ambient.wav")
FOCUS_PATH = os.path.join(AUDIO_DIR, "focus_synthwave.wav")

# sanity check because i keep moving folders around
if not os.path.exists(AMBIENT_PATH) or not os.path.exists(FOCUS_PATH):
    raise FileNotFoundError(f"Missing audio assets! Did you delete the {AUDIO_DIR} folder again?")

# set up the research data logger
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)

# tag every file with the exact time so I don't overwrite my good data
session_time = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = os.path.join(LOG_DIR, f"brain_data_{session_time}.csv")

# write the headers
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "Alpha_Power", "Beta_Power", "Raw_Ratio", "Smoothed_Ratio", "Audio_State"])

# rolling buffers to hold the live brainwaves without eating all my RAM
live_data_buffer = [collections.deque(maxlen=target_buffer_points) for _ in range(4)]
window_size = 5
ratio_window = collections.deque(maxlen=window_size)

# keeping the last 30 ticks for the UI graph so it looks like a smooth wave
plot_x_data = list(range(30))
plot_y_data = collections.deque([0.0]*30, maxlen=30)


def background_music_generator():
    """runs in the background and crossfades the audio based on my current brain state"""
    global current_state, system_running
    
    # start chill
    pygame.mixer.music.load(AMBIENT_PATH)
    pygame.mixer.music.play(-1)
    active_playing_state = "ambient"
    
    while system_running:
        target_state = current_state
        
        # if my brain state changed, swap the music
        if target_state != active_playing_state:
            # nice 1.5s fade out so it doesn't jump scare me
            pygame.mixer.music.fadeout(1500)
            time.sleep(1.5)  
            
            if target_state == "focus":
                pygame.mixer.music.load(FOCUS_PATH)
            else:
                pygame.mixer.music.load(AMBIENT_PATH)
                
            pygame.mixer.music.play(-1)  
            active_playing_state = target_state
            
        time.sleep(0.2)


def eeg_processing_worker():
    """the heavy lifter. pulls the lsl stream and does all the math."""
    global current_state, system_running, plot_y_data
    
    print("--- looking for the muse stream ---")
    streams = resolve_byprop('type', 'EEG', timeout=3.0)
    if len(streams) == 0:
        print("❌ Couldn't find the headset. Did you open BlueMuse?")
        system_running = False
        return

    inlet = StreamInlet(streams[0])
    print(f"📡 Hooked in! Stream: {streams[0].name()}")
    print("\n--- filling the 5-second buffer, hold still... ---")

    while system_running:
        chunk, timestamps = inlet.pull_chunk(timeout=0.0, max_samples=100)
        if chunk:
            for sample in chunk:
                for i in range(4):
                    live_data_buffer[i].append(sample[i])
                    
        # don't do math until we actually have 5 seconds of data
        if len(live_data_buffer[0]) < target_buffer_points:
            time.sleep(0.1)
            continue
            
        # grab the front two sensors (forehead) and average them to get the frontal lobe data
        af7_data = np.array(live_data_buffer[1])
        af8_data = np.array(live_data_buffer[2])
        frontal_signal = (af7_data + af8_data) / 2.0
        
        # mne needs this formatting to run the fast fourier transform
        info = mne.create_info(ch_names=['Frontal'], sfreq=sfreq, ch_types=['eeg'])
        raw_chunk = mne.io.RawArray(frontal_signal.reshape(1, -1), info, verbose=False)
        spectrum = raw_chunk.compute_psd(fmin=1, fmax=40, verbose=False)
        psds, freqs = spectrum.get_data(return_freqs=True)
        
        # isolate alpha (chill) and beta (focus) bands
        alpha_mask = (freqs >= 8) & (freqs <= 13)
        beta_mask = (freqs >= 13) & (freqs <= 30)
        alpha_power = psds[0][alpha_mask].mean()
        beta_power = psds[0][beta_mask].mean()
        
        # the magic number
        raw_ratio = beta_power / alpha_power
        
        # smooth it out so a single eye blink doesn't trigger the synthwave
        ratio_window.append(raw_ratio)
        smoothed_ratio = np.mean(ratio_window)
        
        # HYSTERESIS LOGIC (Calibrated to clean hardware baseline)
        # Require a solid push above the 0.4 resting state to trigger focus
        if current_state == "ambient" and smoothed_ratio > 0.55:
            current_state = "focus"
            
        # Don't drop the focus track until the ratio falls below the resting baseline
        elif current_state == "focus" and smoothed_ratio < 0.35:
            current_state = "ambient"
            
        # Ratios between 0.35 and 0.55 maintain the current audio state
        
        # push the latest ratio to the live graph
        plot_y_data.append(smoothed_ratio)
        
        # log it for the research paper later
        current_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        with open(csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([current_time, f"{alpha_power:.4f}", f"{beta_power:.4f}", f"{raw_ratio:.4f}", f"{smoothed_ratio:.4f}", current_state])
            
        time.sleep(2.0)


# GUI DASHBOARD
class LiveBrainPlotWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Brain-Music BCI Dashboard")
        self.resize(800, 400)
        
        # central widget setup
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QtWidgets.QVBoxLayout(self.central_widget)
        
        # making it dark mode because we aren't savages
        self.graph_widget = pg.PlotWidget()
        self.graph_widget.setBackground('#121212')
        layout.addWidget(self.graph_widget)
        
        # axes labels
        self.graph_widget.setLabel('left', 'Focus Index (Beta/Alpha)', colors='#cccccc', size='14pt')
        self.graph_widget.setLabel('bottom', 'Time (Rolling Window)', colors='#cccccc', size='14pt')
        
        # updated y-range to better fit the new 0.25 to 0.55 baseline
        self.graph_widget.setYRange(0, 0.8)
        self.graph_widget.showGrid(x=True, y=True, alpha=0.2)
        
        # the live brainwave line (neon cyan)
        pen_wave = pg.mkPen(color='#00d2ff', width=3) 
        self.data_line = self.graph_widget.plot(plot_x_data, list(plot_y_data), pen=pen_wave)
        
        # the target line to beat (upper limit - red dashed)
        pen_threshold_high = pg.mkPen(color='#ff3333', width=2, style=QtCore.Qt.DashLine)
        self.threshold_line_high = pg.InfiniteLine(pos=0.55, angle=0, pen=pen_threshold_high)
        self.graph_widget.addItem(self.threshold_line_high)

        # the dropout line (lower limit - yellow dashed)
        pen_threshold_low = pg.mkPen(color='#ffcc00', width=2, style=QtCore.Qt.DashLine)
        self.threshold_line_low = pg.InfiniteLine(pos=0.35, angle=0, pen=pen_threshold_low)
        self.graph_widget.addItem(self.threshold_line_low)
        
        # refresh the graph 10 times a second
        self.timer = QtCore.QTimer()
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()
        
    def update_plot(self):
        # grab the latest data from the eeg thread
        self.data_line.setData(plot_x_data, list(plot_y_data))
        
        # update the window title so I can tell what state I'm in
        self.setWindowTitle(f"EEG Dashboard | Current State: {current_state.upper()}")


# run the whole thing
if __name__ == "__main__":
    # spin up the background threads
    bg_audio = threading.Thread(target=background_music_generator, daemon=True)
    bg_audio.start()
    
    bg_eeg = threading.Thread(target=eeg_processing_worker, daemon=True)
    bg_eeg.start()
    
    # run the GUI on the main thread so it doesn't crash the OS
    app = QtWidgets.QApplication(sys.argv)
    main_window = LiveBrainPlotWindow()
    main_window.show()
    
    # cleanly shut everything down when I close the window
    sys.exit(app.exec_())
    system_running = False
    pygame.mixer.music.stop()