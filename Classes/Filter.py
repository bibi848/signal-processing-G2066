from scipy.signal.windows import tukey
import numpy as np

def filter_signal(signal, dt, f_start, f_end, filter_alpha, 
                  hanning_bool):
    signal = signal - np.mean(signal)
    N = len(signal)

    if hanning_bool:
        window = np.hanning(N)
        signal = signal * window

    fft_vals = np.fft.rfft(signal)
    freqs    = np.fft.rfftfreq(N, dt) / 1e6

    mask = (freqs >= f_start) & (freqs <= f_end)
    indices = np.where(mask)[0]

    if len(indices) > 0:
        filter_window = tukey(len(indices), alpha=filter_alpha)
        
        bp_filter = np.zeros_like(fft_vals, dtype=float)
        bp_filter[indices] = filter_window

        fft_filtered = fft_vals * bp_filter
        return np.fft.irfft(fft_filtered, n=N)
    else:
        return signal

