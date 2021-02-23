# EXPERIMENTAL

# Music Genre Classifier
This is a simple program for predicting the genre of a music.

Still in alpha stage, so expect quite a bit of bugs and inaccurate results.

## The Model
The model currently uses a perceptron model for training the data, will use CNN to get more accurate results (especially for similar genres like pop and rock).

The model is trained using the GTZAN Music Genre Dataset (https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification). The audio tracks are splitted into smaller 3 second parts to have a larger dataset to efficiently train the model.


## How to Use
As the program is still in alpha stage, you have to run the program through a Python IDE by running Main.py (will add an executable later). After running the file, simply click "Choose a Song" button and pick the song of your choice. Currently only .wav files are supported for compatibility with Librosa.
