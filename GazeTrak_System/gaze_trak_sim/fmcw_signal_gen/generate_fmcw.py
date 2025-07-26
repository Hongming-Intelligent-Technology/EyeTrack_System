import numpy as np
import sounddevice as sd
import os

def generate_fmcw(
    f_start=18000,        # Start frequency (Hz)
    f_end=21000,          # End frequency (Hz)
    duration=0.012,       # Duration per frame (seconds)
    fs=50000              # Sampling rate (Hz)
):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    k = (f_end - f_start) / duration
    chirp = np.sin(2 * np.pi * (f_start * t + 0.5 * k * t ** 2))
    return chirp.astype(np.float32)

def save_fmcw(signal, filename='dataset/fmcw.npy'):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.save(filename, signal)
    print(f"FMCW signal saved to {filename}")

def play_fmcw(signal, fs=50000):
    print("Playing FMCW signal (usually inaudible to human ears)...")
    sd.play(signal, samplerate=fs)
    sd.wait()

if __name__ == '__main__':
    fmcw = generate_fmcw()
    save_fmcw(fmcw)

    # Optional playback
    play = input("Play the generated FMCW? [y/N] ").strip().lower()
    if play == 'y':
        play_fmcw(fmcw)
