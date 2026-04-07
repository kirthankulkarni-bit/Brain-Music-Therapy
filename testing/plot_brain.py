import mne
import matplotlib.pyplot as plt

# 1. Load a real sample dataset (built into MNE)
data_path = mne.datasets.sample.data_path()
raw_fname = data_path / "MEG" / "sample" / "sample_audvis_raw.fif"
raw = mne.io.read_raw_fif(raw_fname, preload=True, verbose=False)

# 2. Pick just one EEG channel to keep it simple
raw.pick_types(eeg=True).crop(tmax=60) 

print("--- Calculating Brain Frequencies ---")

# 3. Calculate Power Spectral Density (The EE 'Spectrum Analyzer')
# This turns the time-wave into a frequency graph
spectrum = raw.compute_psd(fmin=1, fmax=40)

# 4. Plot it!
fig = spectrum.plot()
plt.show()