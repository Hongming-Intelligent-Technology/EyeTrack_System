import numpy as np
import tkinter as tk
import time

def show_gaze(
    label_path='dataset/label.npy',
    pred_path='dataset/pred.npy',
    screen_width=1280,
    screen_height=720,
    frame_interval=0.05,   # Duration per frame (seconds)
    draw_mode='dot'        # Options: 'dot' or 'trace'
):
    # 1. Load data
    labels = np.load(label_path)     # [N, 2]
    preds = np.load(pred_path)       # [N, 2]
    assert len(labels) == len(preds), "Mismatch in number of labels and predictions"

    # 2. Initialize window
    root = tk.Tk()
    root.title("Gaze Visualization")
    root.geometry(f"{screen_width}x{screen_height}")
    canvas = tk.Canvas(root, width=screen_width, height=screen_height, bg='white')
    canvas.pack()

    # 3. Play frame sequence
    def draw_frame(i):
        canvas.delete("all")
        x_gt = int(labels[i][0] * screen_width)
        y_gt = int(labels[i][1] * screen_height)
        x_pred = int(preds[i][0] * screen_width)
        y_pred = int(preds[i][1] * screen_height)

        if draw_mode == 'dot':
            # Ground truth - red dot
            canvas.create_oval(x_gt - 5, y_gt - 5, x_gt + 5, y_gt + 5, fill='red')
            # Predicted - blue circle
            canvas.create_oval(x_pred - 5, y_pred - 5, x_pred + 5, y_pred + 5, outline='blue', width=2)
        elif draw_mode == 'trace':
            if i > 0:
                x0_gt = int(labels[i - 1][0] * screen_width)
                y0_gt = int(labels[i - 1][1] * screen_height)
                x0_pred = int(preds[i - 1][0] * screen_width)
                y0_pred = int(preds[i - 1][1] * screen_height)

                canvas.create_line(x0_gt, y0_gt, x_gt, y_gt, fill='red', width=2)
                canvas.create_line(x0_pred, y0_pred, x_pred, y_pred, fill='blue', width=2)

        root.update()
        time.sleep(frame_interval)

    for i in range(len(labels)):
        draw_frame(i)

    print("Playback finished, closing window.")
    root.mainloop()

if __name__ == '__main__':
    show_gaze()
