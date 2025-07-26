import os

from fmcw_signal_gen.generate_fmcw import generate_fmcw, save_fmcw
from audio_capture.record_audio import collect_and_save_audio
from echo_processing.echo_profile import process_audio
from GazeTrak_System.gaze_trak_sim.model_training.predict_lstm import train_model
from GazeTrak_System.gaze_trak_sim.model_training.train_lstm import predict
from visualizer.show_gaze import show_gaze
import numpy as np

def run_pipeline():
    dataset_dir = 'dataset/'

    print("Step 1: Generate FMCW signal")
    fmcw = generate_fmcw()
    save_fmcw(fmcw, filename=os.path.join(dataset_dir, 'fmcw.npy'))

    print("Step 2: Record audio and generate gaze labels")
    collect_and_save_audio(duration=10.0, channels=8)

    print("Step 3: Construct Echo Profile features")
    process_audio()

    print("Step 4: Train gaze model")
    train_model()

    print("Step 5: Run model inference")
    y_pred = predict()
    np.save(os.path.join(dataset_dir, 'pred.npy'), y_pred)

    print("Step 6: Visualize gaze")
    show_gaze(
        label_path=os.path.join(dataset_dir, 'label.npy'),
        pred_path=os.path.join(dataset_dir, 'pred.npy'),
        draw_mode='dot'
    )

    print("Pipeline completed successfully")

if __name__ == '__main__':
    run_pipeline()
