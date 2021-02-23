import numpy as np
import pandas as pd
import librosa
import librosa.display
import json
import os

genres = ["Blues", "Classical", "Country", "Disco", "Hiphop", "Jazz", "Metal", "Pop", "Reggae", "Rock"]

FRAME_SIZE = 1024
HOP_LENGTH = 512  # Used for overlapping frames
SAMPLE_RATE = 22050
SEGMENT_COUNT = 10  # Dividing the audio into parts so that we get more data (Original data has 1000 files, not enough to get good results)
EXPECTED_LENGTH = int(int(np.ceil(SAMPLE_RATE * 30 / HOP_LENGTH)) / SEGMENT_COUNT) + 1 #For 3 seconds of audio (10 segments for 30 seconds)


def getSamples(path):
    dataset = {
        "Genre" : [],
        "Root Mean Square Energy": [],
        "Zero-Crossing Rate" : [],
        "Spectral Centroid": [],
        "Spectral Rolloff": [],
        "Spectral Flux": [],
        "Spectral Bandwith": [],
        "Spectral Contrast": [],
        "Spectral Flatness": [],
        # Splitting the MFCC into n_mfcc values (easier to train the data, as the MFCC values will have the same size as the other features)
        "MFCC1": [], "MFCC2": [], "MFCC3": [], "MFCC4": [],
        "MFCC5": [], "MFCC6": [], "MFCC7": [], "MFCC8": [],
        "MFCC9": [], "MFCC10": [], "MFCC11": [], "MFCC12": [], "MFCC13": [],
        "MFCC14": [], "MFCC15": [], "MFCC16": [], "MFCC17": [], "MFCC18": [], "MFCC19": [], "MFCC20": [],
    }

    for dirpath, dirnames, filenames in os.walk(path):
        if dirpath is not path:   #Ignoring the first path, as it's the genres folder itself with its subfolders
            for i in filenames:
                path = dirpath + '\\' + i
                audio, sr = librosa.load(path, sr=SAMPLE_RATE)

                # Feature Extraction - Getting the appropriate features from the audio data to train our model

                for j in range(SEGMENT_COUNT):
                    start = j * int(SAMPLE_RATE * 30 / SEGMENT_COUNT)
                    end = (j + 1) * int(SAMPLE_RATE * 30 / SEGMENT_COUNT)

                    # For some of those features, their shape is (1, value), so we only need the first element of the 2D-array (which is also the only element)
                    rootMeanSquareEnergy = librosa.feature.rms(audio[start:end], frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
                    zeroCrossingRate = librosa.feature.zero_crossing_rate(audio[start:end], frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
                    spectralCentroid = librosa.feature.spectral_centroid(audio[start:end], sr=SAMPLE_RATE, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
                    spectralRolloff = librosa.feature.spectral_rolloff(audio[start:end], sr=SAMPLE_RATE, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
                    spectralFlux = librosa.onset.onset_strength(audio[start:end], sr=SAMPLE_RATE)
                    spectralBandwith = librosa.feature.spectral_bandwidth(audio[start:end], sr=SAMPLE_RATE, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
                    spectralContrast = librosa.feature.spectral_contrast(audio[start:end], sr=SAMPLE_RATE, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
                    spectralFlatness = librosa.feature.spectral_flatness(audio[start:end], n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
                    MFCC = librosa.feature.mfcc(audio[start:end], n_fft=FRAME_SIZE, hop_length=HOP_LENGTH, n_mfcc=20)  # 20 gave me better results than 13

                    if len(rootMeanSquareEnergy) == EXPECTED_LENGTH:  # Only append the new data if it's size is the expected size
                        dataset["Genre"].append(dirpath.split("\\")[-1])
                        dataset["Root Mean Square Energy"].append(np.mean(rootMeanSquareEnergy).tolist())
                        dataset["Zero-Crossing Rate"].append(np.mean(zeroCrossingRate).tolist())
                        dataset["Spectral Centroid"].append(np.mean(spectralCentroid).tolist())
                        dataset["Spectral Rolloff"].append(np.mean(spectralRolloff).tolist())
                        dataset["Spectral Flux"].append(np.mean(spectralFlux).tolist())
                        dataset["Spectral Bandwith"].append(np.mean(spectralBandwith).tolist())
                        dataset["Spectral Contrast"].append(np.mean(spectralContrast).tolist())
                        dataset["Spectral Flatness"].append(np.mean(spectralFlatness).tolist())
                        for k in range(20):
                            dataset["MFCC" + str(k + 1)].append(np.mean(MFCC[k]).tolist())
                        print(path)

    return dataset

if __name__ == "__main__":
    dataset = getSamples("genres")
    with open("MusicalDataset.json", "w") as fp:
        json.dump(dataset, fp)
