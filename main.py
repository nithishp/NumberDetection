import tkinter as tk
import joblib
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

class App:
    def __init__(self, master):
        self.master = master
        master.title("Digit Recognizer")

        self.modelpath = 'hadwittennumbers\digits.joblib'
        self.model = joblib.load(self.modelpath)

        self.canvas = tk.Canvas(master, width=400, height=400)
        self.canvas.pack()

        self.button = tk.Button(master, text="Recognize", command=self.recognize)
        self.button.pack()

        master.protocol("WM_DELETE_WINDOW", self.close)

    def recognize(self):
        self.canvas.delete("all")
        for i in range(10):
            plt.subplot(2, 5, i + 1)
            plt.imshow(load_digits().images[i], cmap='gray')
            plt.title(self.model.predict(load_digits().data[i:i+1]))
            plt.axis('off')
            plt.savefig(f"digit_{i}.png")
            photo = tk.PhotoImage(file=f"digit_{i}.png")
            self.canvas.create_image(0, 0, anchor="nw", image=photo)
            self.canvas.image = photo

    def close(self):
        self.master.quit()

root = tk.Tk()
app = App(root)
root.mainloop()