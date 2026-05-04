import pickle
import numpy as np

# Load participant 1
with open('s01.dat', 'rb') as f:
    # Use encoding='latin1' because the files were created in Python 2
    raw_data = pickle.load(f, encoding='latin1')

eeg_data = raw_data['data'] # This is (40, 40, 8064)
labels = raw_data['labels']

# EEG is the first 32 channels (the rest are peripheral like heart rate)
just_eeg = eeg_data[:, :32, :] 

print(f"Loaded {just_eeg.shape[0]} trials of EEG data.")
print(f"Sampling Rate: 128Hz")