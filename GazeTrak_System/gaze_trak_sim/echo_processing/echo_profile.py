import numpy as np
from scipy.signal import butter, lfilter, correlate
import os

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low, high = lowcut / nyq, highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_filter(data, lowcut=18000, highcut=21000, fs=50000):
    b, a = butter_bandpass(lowcut, highcut, fs)
    return lfilter(b, a, data, axis=0)

def cross_correlate_all_channels(audio, fmcw):
    """Perform cross-correlation for each channel, output shape = [frames, channels, correlation_window]"""
    num_channels = audio.shape[1]
    window_size = len(fmcw)
    corr_list = []

    for ch in range(num_channels):
        ch_data = audio[:, ch]
        corr = correlate(ch_data, fmcw, mode='valid')  # [samples - N + 1]
        corr_list.append(corr)

    # Stack → [channels, time]
    return np.stack(corr_list, axis=0)  # [C, T]

def slice_echo_profile(corr, frame_len, hop_len, frame_num=26, frame_width=60):
    """
    Slice long-time echo data into model input frame segments:
    Input shape: [C, T] → Output: [N, C, 26, 60]
    """
    C, T = corr.shape
    features = []
    start = 0
    while start + frame_num * hop_len <= T:
        frame = []
        for c in range(C):
            spec = []
            for i in range(frame_num):
                segment = corr[c, start + i * hop_len : start + i * hop_len + frame_width]
                spec.append(segment)
            spec = np.stack(spec, axis=0)  # [26, 60]
            frame.append(spec)
        echo = np.stack(frame, axis=0)  # [C, 26, 60]
        features.append(echo)
        start += frame_len  # Sliding window

    return np.stack(features, axis=0)  # [N, C, 26, 60]

def process_audio(
    audio_path='dataset/raw_audio.npy',
    fmcw_path='dataset/fmcw.npy',
    save_dir='dataset/',
    fs=50000
):
    audio = np.load(audio_path)        # [samples, channels]
    fmcw = np.load(fmcw_path)          # [samples]
    os.makedirs(save_dir, exist_ok=True)

    print(f"Loaded audio shape = {audio.shape}, FMCW shape = {fmcw.shape}")

    filtered = apply_filter(audio, fs=fs)               # [samples, channels]
    corr = cross_correlate_all_channels(filtered, fmcw) # [channels, time]
    echo_features = slice_echo_profile(
        corr,
        frame_len=int(fs * 0.012),   # Frame duration 12ms
        hop_len=5,                   # Sliding within frame
        frame_num=26,
        frame_width=60
    )                                # [N, 16, 26, 60]

    print(f"Echo Profile shape = {echo_features.shape}")
    np.save(os.path.join(save_dir, 'train.npy'), echo_features)

    # Synchronize and trim labels
    labels = np.load(os.path.join(save_dir, 'label.npy'))
    labels = labels[:len(echo_features)]
    np.save(os.path.join(save_dir, 'label.npy'), labels)
    print(f"Saved Echo features and labels, sample count = {len(labels)}")

if __name__ == '__main__':
    process_audio()
