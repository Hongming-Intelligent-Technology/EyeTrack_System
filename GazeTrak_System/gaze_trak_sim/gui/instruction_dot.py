import tkinter as tk
import numpy as np
import time
import threading
import random

class InstructionDotApp:
    def __init__(self, duration, refresh_rate=1/0.012):
        self.duration = duration
        self.interval = int(1000 * refresh_rate)  # ms
        self.positions = []

        self.root = tk.Tk()
        self.root.attributes('-fullscreen', True)
        self.root.configure(bg='white')
        self.canvas = tk.Canvas(self.root, bg='white')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.dot = None
        self.running = True

        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()

    def start(self):
        total_frames = int(self.duration / (self.interval / 1000))
        self.update_frame(total_frames)
        self.root.after(int(self.duration * 1000), self.stop)
        self.root.mainloop()

    def update_frame(self, frames_left):
        if not self.running or frames_left <= 0:
            return
        # Generate random position (can be replaced with grid traversal)
        x = random.randint(100, self.screen_width - 100)
        y = random.randint(100, self.screen_height - 100)
        self.positions.append([x, y])

        # Clear old dot and draw new dot
        self.canvas.delete("dot")
        radius = 10
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill='red', tags="dot")

        self.root.after(self.interval, self.update_frame, frames_left - 1)

    def stop(self):
        self.running = False
        self.root.destroy()

def run_instruction_dot(total_duration=10.0, refresh_rate=1/0.012):
    app = InstructionDotApp(duration=total_duration, refresh_rate=refresh_rate)
    
    # Run GUI in main thread
    gui_thread = threading.Thread(target=app.start)
    gui_thread.start()
    gui_thread.join()

    # Normalize to [0, 1]
    labels = np.array(app.positions, dtype=np.float32)
    labels[:, 0] /= app.screen_width
    labels[:, 1] /= app.screen_height
    return labels

if __name__ == '__main__':
    labels = run_instruction_dot(5.0, refresh_rate=1/0.012)
    print("Collected gaze label:", labels.shape)
