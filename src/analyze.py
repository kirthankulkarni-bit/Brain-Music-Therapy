import numpy as np
from scipy.signal import welch

def get_band_powers(data, fs=256):
    """
    fs: sampling rate (Muse is typically 256Hz)
    """
    # 1. Estimate Power Spectral Density (PSD)
    # Using a shorter window (nperseg) helps with 1-second chunks
    freqs, psd = welch(data, fs=fs, nperseg=len(data))
    
    # 2. Extract specific frequency bands
    # Alpha: 8-12 Hz | Beta: 13-30 Hz
    alpha_idx = np.logical_and(freqs >= 8, freqs <= 12)
    beta_idx = np.logical_and(freqs >= 13, freqs <= 30)
    
    # 3. Sum the power
    alpha_power = np.sum(psd[alpha_idx])
    beta_power = np.sum(psd[beta_idx])
    
    return alpha_power, beta_power