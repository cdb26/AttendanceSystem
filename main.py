import tkinter as tk
from pathlib import Path

from functions.train import trainImg
from assets.designs.centerWindow import center_window

BASE_DIR = Path(__file__).resolve().parent
photo_path = BASE_DIR / "assets" / "pics" / "icon.png"

root = tk.Tk()
width = 1320
height = 850
root.title("Attendance System")
photo = tk.PhotoImage(file=str(photo_path))
root.iconphoto(False, photo)

center_window(root, width, height)

root.mainloop()