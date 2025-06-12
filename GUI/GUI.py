import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
from skimage.transform import resize, downscale_local_mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading

from utils.trinarisation2 import detour_clean, detour_clean_stack
from utils.propreties4 import bone_fraction, trabecular_thickness, trabecular_distance, anisotropy
import SimpleITK as sitk
from utils.binarisation3 import separate_trabecular_cortical_by_thickness

class BoneViewerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Bone Viewer & Analyzer")
        self.geometry("1200x800")
        self.tiff_image = None
        self.slice_index = 0
        self.slice_count = 0
        self.image_np = None
        self.full_stack = []

        self.start_slice = 0
        self.end_slice = None  # Will be set after loading
        self.create_widgets()

    def create_widgets(self):
        top_frame = tk.Frame(self, bg="white")
        top_frame.pack(fill="x", padx=10, pady=10)

        tk.Button(top_frame, text="Load TIF File", command=self.load_tiff).pack(side="left", padx=5)
        tk.Button(top_frame, text="◀ Previous", command=self.prev_slice).pack(side="left", padx=5)
        tk.Button(top_frame, text="Next ▶", command=self.next_slice).pack(side="left", padx=5)
        self.slice_label = tk.Label(top_frame, text="Slice: 0", bg="white", font=("Arial", 12))
        self.slice_label.pack(side="left", padx=10)

        tk.Button(top_frame, text="Show Histogram", command=self.show_histogram).pack(side="left", padx=5)
        tk.Button(top_frame, text="Analyze BV/TV (slice)", command=self.analyze_slice).pack(side="left", padx=5)
        tk.Button(top_frame, text="Analyze 3D Properties", command=self.analyze_stack).pack(side="left", padx=5)

        self.analysis_text = tk.Label(top_frame, text="", bg="white", font=("Arial", 10), anchor="w", justify="left")
        self.analysis_text.pack(side="left", padx=20)

        tk.Button(top_frame, text="Separate Trabecular & Cortical", command=self.separate_trabecular_cortical).pack(side="left", padx=5)

        self.canvas_frame = tk.Frame(self)
        self.canvas_frame.pack(fill="both", expand=True)

        # Progress bar
        self.progress_label = tk.Label(self, text="", bg="white", fg="green", font=("Arial", 10))
        self.progress_label.pack(pady=2)
        self.progress_bar = ttk.Progressbar(self, orient="horizontal", length=400, mode="determinate")
        self.progress_bar.pack(pady=3)
        # Removed erroneous reference to self.progress
        # self.progress.pack(pady=5)

        # Slider to select slice index
        self.slice_slider = tk.Scale(self, from_=0, to=0, orient="horizontal", label="Navigate Slices", command=self.slider_moved)
        self.slice_slider.pack(fill="x", padx=20, pady=5)

        # Start and End slice input
        range_frame = tk.Frame(self)
        range_frame.pack()
        tk.Label(range_frame, text="Start Slice:").pack(side="left")
        self.start_entry = tk.Entry(range_frame, width=5)
        self.start_entry.pack(side="left", padx=5)
        tk.Label(range_frame, text="End Slice:").pack(side="left")
        self.end_entry = tk.Entry(range_frame, width=5)
        self.end_entry.pack(side="left", padx=5)

        self.fig, self.axs = plt.subplots(1, 2, figsize=(12, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def load_tiff(self):
        self.slice_slider.config(state="normal")
        file_path = filedialog.askopenfilename(filetypes=[("TIFF files", "*.tif")])
        if not file_path:
            return
        self.tiff_image = Image.open(file_path)
        self.slice_count = self.tiff_image.n_frames
        self.slice_index = 0
        self.full_stack = []
        for i in range(self.slice_count):
            self.tiff_image.seek(i)
            img = np.array(self.tiff_image.convert("L"))
            downscaled = downscale_local_mean(img, (2, 2)).astype(np.uint8)
            self.full_stack.append(downscaled)
        self.slice_slider.config(to=self.slice_count - 1)
        self.slice_slider.set(self.slice_index)
        self.update_display()

    def prev_slice(self):
        if self.tiff_image and self.slice_index > 0:
            self.slice_index -= 1
            self.update_display()

    def next_slice(self):
        if self.tiff_image and self.slice_index < self.slice_count - 1:
            self.slice_index += 1
            self.update_display()

    def slider_moved(self, val):
        self.slice_index = int(val)
        self.update_display()

    def update_display(self):
        self.progress_label.config(text="Rendering slice...")
        self.progress_bar.start(10)
        self.update_idletasks()
        self.tiff_image.seek(self.slice_index)
        img = np.array(self.tiff_image.convert("L"))
        self.image_np = downscale_local_mean(img, (2, 2)).astype(np.uint8)
        self.slice_label.config(text=f"Slice: {self.slice_index + 1}/{self.slice_count}")

        segmented = detour_clean(self.image_np)

        self.axs[0].clear()
        display_img = resize(self.image_np, (300, 300), preserve_range=True).astype(np.uint8)
        self.axs[0].imshow(display_img, cmap='gray')
        self.axs[0].set_title("Original Slice")
        self.axs[0].axis('off')

        self.axs[1].clear()
        self.axs[1].imshow(segmented, cmap='nipy_spectral')
        self.axs[1].set_title("Trinarized Mask")
        self.axs[1].axis('off')

        self.canvas.draw()
        self.progress_bar.stop()
        self.progress_label.config(text="")

    def show_histogram(self):
        if self.image_np is None:
            return
        hist, bins = np.histogram(self.image_np, bins=256, range=(0, 256))

        plt.figure("Histogram")
        plt.clf()
        plt.title("Gray Level Histogram")
        plt.xlabel("Gray Level")
        plt.ylabel("Number of Pixels")
        plt.bar(bins[:-1], hist, width=np.diff(bins), edgecolor="black")
        plt.show()

    def analyze_slice(self):
        if self.image_np is None:
            return

        segmented = detour_clean(self.image_np)
        trab = np.where(segmented == 1, 1, 0)  # Assuming label 1 is trabecular

        bvtv = bone_fraction(segmented)
        result_text = f"BV/TV (slice {self.slice_index + 1}): {bvtv:.3f}"
        self.analysis_text.config(text=result_text)
        self.show_result_popup(result_text)

    def separate_trabecular_cortical(self):
        if not self.full_stack:
            return

        self.progress.config(text="Separating trabecular and cortical regions...")
        self.update_idletasks()

        # Step 1: use already trinarized full stack (label 2 = bone)
        start = int(self.start_entry.get()) if self.start_entry.get().isdigit() else 0
        end = int(self.end_entry.get()) if self.end_entry.get().isdigit() else len(self.full_stack)
        sliced_stack = self.full_stack[start:end]
        segmented_stack = np.stack([detour_clean(slice_img) for slice_img in sliced_stack])

        # Step 2: apply separation based on thickness
        bone_mask = (segmented_stack == 2).astype(np.uint8)
        image = sitk.GetImageFromArray(bone_mask)
        distance_map = sitk.SignedMaurerDistanceMap(image, insideIsPositive=True, squaredDistance=False)
        thickness_map = sitk.GetArrayFromImage(distance_map) * 2

        label_mask = separate_trabecular_cortical_by_thickness(segmented_stack, threshold_px=80)

        # Step 3: show the current slice in color
        label_slice = label_mask[self.slice_index]

        self.axs[1].clear()
        self.axs[1].imshow(label_slice, cmap='nipy_spectral')
        self.axs[1].set_title("Trabecular (2) / Cortical (3)")
        self.axs[1].axis('off')

        self.canvas.draw()
        self.progress_bar.stop()
        self.progress_label.config(text="")


        def show_result_popup(self, result_text):
            result_window = tk.Toplevel(self)
            result_window.title("Analysis Result")
            result_label = tk.Label(result_window, text=result_text, font=("Arial", 12), justify="left")
            result_label.pack(padx=20, pady=20)

    def analyze_stack(self):
        def run_analysis():
            self.progress_label.config(text="Analyzing full volume...")
            self.progress_bar.start(10)
            self.update_idletasks()
            if not self.full_stack:
                return

            start = int(self.start_entry.get()) if self.start_entry.get().isdigit() else 0
            end = int(self.end_entry.get()) if self.end_entry.get().isdigit() else len(self.full_stack)
            sliced_stack = self.full_stack[start:end]
            segmented_stack = detour_clean_stack(np.stack(sliced_stack))
            trab_volume = np.where(segmented_stack == 1, 1, 0)

            tbth = trabecular_thickness(segmented_stack)
            tbsp = trabecular_distance(segmented_stack)
            anisotropy_val = anisotropy("TEMP_FILE_STACK.tif")  # placeholder: save segmented_stack to file first

            bvtv = bone_fraction(segmented_stack)
            result_text = f"3D Analysis (whole volume): BV/TV: {bvtv:.3f} Tb.Th: {tbth:.2f} Tb.Sp: {tbsp:.2f} Anisotropy: {anisotropy_val:.2f}"
            self.progress.config(text="")
            self.analysis_text.config(text=result_text)

        threading.Thread(target=run_analysis).start()

if __name__ == "__main__":
    app = BoneViewerApp()
    app.mainloop()
