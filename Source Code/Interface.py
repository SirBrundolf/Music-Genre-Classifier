from CreateDataset import genres
from TrainData import predictGenre
from joblib import load
from tensorflow.keras.models import load_model
from tkinter.filedialog import askopenfilename
import tkinter as tk

WIDTH = 960
HEIGHT = 600

labels = []

def setupMenu():
    root = tk.Tk()
    root.title('Music Genre Classifier')
    root.resizable(width=False, height=False)

    canvas = tk.Canvas(root, height=HEIGHT, width=WIDTH)
    canvas.pack()

    title = tk.Label(root, text='Music Genre Classifier', font=("Courier", 35))
    title.place(x=170, y=100)
    labels.append(title)

    start = tk.Button(root, text='Choose a Song (.wav only):', height=2, width=40, font=("Courier", 15), command=lambda : predictTrack())
    exit = tk.Button(root, text='Exit', height=2, width=40, font=("Courier", 15), command=root.destroy)
    start.place(x=240, y=340)
    exit.place(x=240, y=400)

    root.mainloop()
    return root


def predictTrack():
    tk.Tk().withdraw()
    filename = askopenfilename()

    if filename == "" :   #If choosing close instead of opening a file
        return

    for label in labels:
        label.destroy()

    print("Loading the model.")
    model = load_model('MusicGenreClassifierModel.h5')

    print("Loading the scaler.")
    scaler = load('StandardScaler.bin')

    print("Loading the file and making predictions.")
    predictedGenre = predictGenre(filename, model, scaler)

    print("Predicted genre(s) for", filename.split("/")[-1] + ": ", end="")

    for i in predictedGenre:
        print(genres[i], end="")
        if i != predictedGenre[-1]:
            print(", ", end="")
    print()