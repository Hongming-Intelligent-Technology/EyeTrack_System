import sounddevice as sd
import numpy as np
import threading
import time
import os
from fmcw_signal_gen.generate_fmcw import generate_fmcw
from gui.instruction_dot import run_instruction_dot

def collect_and_save_audio(
    duration=10.0,        # Total recording duration (seconds)
    fs=50000,             # Sampling rate
    channels=8,           # Number of microphone channels
    frame_duration=0.012, # Single frame FMCW duration (seconds)
    save_dir='dataset/'
):
    os.makedirs(save_dir, exist_ok=True)
    print(f"Recording {duration}s @ {fs}Hz, channels = {channels}")

    # 1. Generate FMCW signal
    fmcw = generate_fmcw(duration=frame_duration, fs=fs)
    fmcw_repeats = int(duration / frame_duration)
    tx_signal = np.tile(fmcw, fmcw_repeats)

    # 2. Launch red dot GUI (to get gaze label)
    label_container = []

    def gui_thread():
        labels = run_instruction_dot(duration, refresh_rate=1/frame_duration)
        label_container.append(labels)

    t_gui = threading.Thread(target=gui_thread)
    t_gui.start()
    time.sleep(0.2)  # Wait for GUI to start

    # 3. Play FMCW + Record audio
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=channels, dtype='float32')
    sd.play(tx_signal, samplerate=fs)
    sd.wait()  # Recording ends after playback

    # 4. Save data
    np.save(os.path.join(save_dir, 'raw_audio.npy'), recording)
    print(f"Saved audio to raw_audio.npy, shape={recording.shape}")

    t_gui.join()  # Wait for GUI to finish
    labels = label_container[0]
    np.save(os.path.join(save_dir, 'label.npy'), labels)
    print(f"Saved gaze label to label.npy, shape={labels.shape}")

if __name__ == '__main__':
    collect_and_save_audio()
