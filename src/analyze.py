import numpy as np
from scipy.signal import welch

def calculate_focus_index(alpha_power, beta_power):
    """
    Calculates a focus ratio. 
    More Beta = More Focus. More Alpha = More Relaxation.
    """
    if alpha_power == 0:
        return 0
    
    # Simple Beta/Alpha ratio
    ratio = beta_power / alpha_power
    
    # Optional: Clamp the value between 0 and 5 for stability
    return max(0, min(ratio, 5))

def get_musical_parameters(focus_index):
    """
    Translates the Focus Index into numbers a music AI can understand.
    """
    # 0.0 to 1.0 scale: 0 is 'Chill/Ambient', 1 is 'Energetic/Complex'
    intensity = focus_index / 2.0 
    return max(0, min(intensity, 1.0))

def get_band_powers(data, fs=256):
    """
    Extracts Alpha and Beta power using Welch's method.
    """
    freqs, psd = welch(data, fs, nperseg=fs)
    
    # Define bands
    alpha_idx = np.where((freqs >= 8) & (freqs <= 12))
    beta_idx = np.where((freqs >= 13) & (freqs <= 30))
    
    alpha_power = np.mean(psd[alpha_idx])
    beta_power = np.mean(psd[beta_idx])
    
    return alpha_power, beta_power