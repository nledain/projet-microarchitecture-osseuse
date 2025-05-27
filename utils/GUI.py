import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

class LanguageApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Language Practice")
        self.geometry("800x500")
        self.configure(bg="white")
        self.resizable(True, True)

        # Configure the main grid
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        self.create_widgets()

    def create_widgets(self):
        # Top bar frame
        top_frame = tk.Frame(self, bg="white")
        top_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        top_frame.columnconfigure(1, weight=1)

        # Flag image
        flag_img = Image.open("C:/Users/20109/Desktop/Flag_of_Spain.png").resize((50, 30))
        self.flag_photo = ImageTk.PhotoImage(flag_img)
        flag_label = tk.Label(top_frame, image=self.flag_photo, bg="white")
        flag_label.grid(row=0, column=0, sticky="w")

        # Progress bar canvas
        progress_canvas = tk.Canvas(top_frame, width=400, height=20, bg="white", highlightthickness=0)
        progress_canvas.grid(row=0, column=1)
        progress_canvas.create_rectangle(2, 2, 400, 20, outline="black", width=1)
        progress_canvas.create_rectangle(2, 2, 0.73 * 400, 20, fill="#f68b1e", outline="")

        # Progress %
        progress_label = tk.Label(top_frame, text="73%", fg="#f68b1e", font=("Arial", 10, "bold"), bg="white")
        progress_label.grid(row=0, column=2, padx=10)

        # Close button
        close_btn = tk.Button(top_frame, text="✖", bg="white", bd=0, font=("Arial", 14), command=self.destroy)
        close_btn.grid(row=0, column=3, sticky="e")

        # Main content frame
        main_frame = tk.Frame(self, bg="white")
        main_frame.grid(row=1, column=0, sticky="nsew", padx=20)

        tk.Label(main_frame, text="Translate the word in English:", font=("Comic Sans MS", 14), bg="white").pack(pady=(40, 0))
        tk.Label(main_frame, text="Aguacate", font=("Comic Sans MS", 28, "bold"), bg="white").pack(pady=(10, 30))

        # Answer entry area
        entry_frame = tk.Frame(main_frame, highlightbackground="black", highlightthickness=2, bg="white")
        entry_frame.pack(pady=10)
        self.answer_entry = tk.Entry(entry_frame, font=("Comic Sans MS", 14), width=30, bd=0)
        self.answer_entry.insert(0, "Type your answer here...")
        self.answer_entry.bind("<FocusIn>", self.clear_placeholder)
        self.answer_entry.pack(padx=20, pady=20)

    def clear_placeholder(self, event):
        if self.answer_entry.get() == "Type your answer here...":
            self.answer_entry.delete(0, tk.END)

if __name__ == "__main__":
    app = LanguageApp()
    app.mainloop()

