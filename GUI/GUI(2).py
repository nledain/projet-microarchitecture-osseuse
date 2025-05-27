import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from utils.segmentation import detour_clean
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class BoneViewerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Bone Viewer & Analyzer")
        self.geometry("1000x700")
        self.tiff_image = None
        self.slice_index = 0
        self.slice_count = 0
        self.image_np = None

        self.create_widgets()

    def create_widgets(self):
        top_frame = tk.Frame(self, bg="white")
        top_frame.pack(fill="x", padx=10, pady=10)

        load_btn = tk.Button(top_frame, text="Load TIF File", command=self.load_tiff)
        load_btn.pack(side="left", padx=5)

        prev_btn = tk.Button(top_frame, text="◀ Previous", command=self.prev_slice)
        prev_btn.pack(side="left", padx=5)

        next_btn = tk.Button(top_frame, text="Next ▶", command=self.next_slice)
        next_btn.pack(side="left", padx=5)

        self.slice_label = tk.Label(top_frame, text="Slice: 0", bg="white", font=("Arial", 12))
        self.slice_label.pack(side="left", padx=10)

        self.canvas_frame = tk.Frame(self)
        self.canvas_frame.pack(fill="both", expand=True)

        self.fig, self.axs = plt.subplots(1, 2, figsize=(10, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def load_tiff(self):
        file_path = filedialog.askopenfilename(filetypes=[("TIFF files", "*.tif")])
        if not file_path:
            return
        self.tiff_image = Image.open(file_path)
        self.slice_count = self.tiff_image.n_frames
        self.slice_index = 0
        self.update_display()

    def prev_slice(self):
        if self.tiff_image and self.slice_index > 0:
            self.slice_index -= 1
            self.update_display()

    def next_slice(self):
        if self.tiff_image and self.slice_index < self.slice_count - 1:
            self.slice_index += 1
            self.update_display()

    def update_display(self):
        self.tiff_image.seek(self.slice_index)
        self.image_np = np.array(self.tiff_image.convert("L"))

        self.slice_label.config(text=f"Slice: {self.slice_index + 1}/{self.slice_count}")

        segmented = detour_clean(self.image_np)

        self.axs[0].clear()
        self.axs[0].imshow(self.image_np, cmap='gray')
        self.axs[0].set_title("Original Slice")
        self.axs[0].axis('off')

        self.axs[1].clear()
        self.axs[1].imshow(segmented, cmap='nipy_spectral')
        self.axs[1].set_title("Segmented Mask")
        self.axs[1].axis('off')

        self.canvas.draw()

if __name__ == "__main__":
    app = BoneViewerApp()
    app.mainloop()
