import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
from CreateDataset import genres, FRAME_SIZE, HOP_LENGTH, SAMPLE_RATE, SEGMENT_COUNT

if __name__ == "__main__":
    dataset = pd.read_json("MusicalDataset.json")
    #print(dataset.head())
    #exit()

    # To process the "Genre" data, we need to convert its text data to numeric data
    from sklearn.preprocessing import LabelEncoder

    genresDF = dataset.iloc[:, 0]
    encoder = LabelEncoder()
    genresEncoded = encoder.fit_transform(genresDF)
    dataset['Genre'] = genresEncoded


    from sklearn.model_selection import train_test_split

    X = dataset.drop('Genre', axis=1).values
    y = dataset['Genre'].values

    # 60% train, %20 validation, 20% test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25)


    # Normalizing the data
    from sklearn.preprocessing import StandardScaler

    standardScaler = StandardScaler()
    X_train = standardScaler.fit_transform(X_train)
    X_val = standardScaler.transform(X_val)   # Not fitting the test data to gain information about it (prevents data leakage)
    X_test = standardScaler.transform(X_test)

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout

    model = Sequential([
        Dense(512, activation='relu'),
        Dropout(0.7),

        Dense(256, activation='relu'),
        Dropout(0.5),

        Dense(128, activation='relu'),
        Dropout(0.5),

        Dense(64, activation='relu'),
        Dropout(0.5),

        Dense(len(genres), activation='softmax')
    ])

    # Using Early Stopping to prevent overfitting
    from tensorflow.keras.callbacks import EarlyStopping
    earlyStop = EarlyStopping(monitor='val_loss', mode='min', patience=25)

    from tensorflow.keras.optimizers import Adam

    opt = Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
    name='Adam')  #Default learning rate is 0.001

    model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(x=X_train,
            y=y_train,
            epochs=300,
            batch_size=128,
            validation_data=(X_val, y_val),
            callbacks=[earlyStop]
            )


    from sklearn.metrics import classification_report, confusion_matrix

    print(model.evaluate(X_test, y_test))

    predictions = np.argmax(model.predict(X_test), axis=-1)

    X = standardScaler.fit_transform(X)
    predictionsAll = np.argmax(model.predict(X), axis=-1)

    print(classification_report(y_test, predictions))
    print(confusion_matrix(y_test, predictions))

    print(classification_report(y, predictionsAll))
    print(confusion_matrix(y, predictionsAll))

    model.summary()

    from joblib import dump
    dump(standardScaler, 'StandardScaler.bin', compress=True)


    model.save('MusicGenreClassifierModel.h5')

    finishData = pd.DataFrame(history.history)
    fig, axs = plt.subplots(2)
    axs[0].plot(finishData["accuracy"], label="Train Accuracy")
    axs[0].plot(finishData["val_accuracy"], label="Validation Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_yticks(np.arange(0, 1, 0.1))
    axs[1].plot(finishData["loss"], label="Train Error")
    axs[1].plot(finishData["val_loss"], label="Validation Error")
    axs[1].legend(loc="upper right")
    axs[1].set_yticks(np.arange(2, 0, -0.2))
    plt.show()


def predictGenre(path, inputModel, inputScaler):
    genreList = np.zeros((len(genres)), dtype=int)
    track, sr = librosa.load(path, sr=SAMPLE_RATE)

    for i in range(SEGMENT_COUNT * 3):
        start = i * int(SAMPLE_RATE * 30 / (SEGMENT_COUNT * 3))
        end = (i + 1) * int(SAMPLE_RATE * 30 / (SEGMENT_COUNT * 3))

        rootMeanSquareEnergy = librosa.feature.rms(track[start:end], frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
        zeroCrossingRate = librosa.feature.zero_crossing_rate(track[start:end], frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
        spectralCentroid = librosa.feature.spectral_centroid(track[start:end], n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
        spectralRolloff = librosa.feature.spectral_rolloff(track[start:end], n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
        spectralFlux = librosa.onset.onset_strength(track[start:end], sr=SAMPLE_RATE)
        spectralBandwith = librosa.feature.spectral_bandwidth(track[start:end], sr=SAMPLE_RATE, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
        spectralContrast = librosa.feature.spectral_contrast(track[start:end], sr=SAMPLE_RATE, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
        spectralFlatness = librosa.feature.spectral_flatness(track[start:end], n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
        MFCC = librosa.feature.mfcc(track[start:end], n_fft=FRAME_SIZE, hop_length=HOP_LENGTH, n_mfcc=20)

        trackData = {
            "Root Mean Square Energy": [np.mean(rootMeanSquareEnergy)],
            "Zero-Crossing Rate": [np.mean(zeroCrossingRate)],
            "Spectral Centroid": [np.mean(spectralCentroid)],
            "Spectral Rolloff": [np.mean(spectralRolloff)],
            "Spectral Flux": [np.mean(spectralFlux)],
            "Spectral Bandwith": [np.mean(spectralBandwith)],
            "Spectral Contrast": [np.mean(spectralContrast)],
            "Spectral Flatness": [np.mean(spectralFlatness)],
            "MFCC1": [np.mean(MFCC[0])], "MFCC2": [np.mean(MFCC[1])], "MFCC3": [np.mean(MFCC[2])],
            "MFCC4": [np.mean(MFCC[3])], "MFCC5": [np.mean(MFCC[4])], "MFCC6": [np.mean(MFCC[5])],
            "MFCC7": [np.mean(MFCC[6])], "MFCC8": [np.mean(MFCC[7])], "MFCC9": [np.mean(MFCC[8])],
            "MFCC10": [np.mean(MFCC[9])], "MFCC11": [np.mean(MFCC[10])], "MFCC12": [np.mean(MFCC[11])],
            "MFCC13": [np.mean(MFCC[12])], "MFCC14": [np.mean(MFCC[13])], "MFCC15": [np.mean(MFCC[14])],
            "MFCC16": [np.mean(MFCC[15])], "MFCC17": [np.mean(MFCC[16])], "MFCC18": [np.mean(MFCC[17])],
            "MFCC19": [np.mean(MFCC[18])], "MFCC20": [np.mean(MFCC[19])]
        }

        df = pd.DataFrame(trackData)
        scaledData = inputScaler.transform(df.values)
        prediction = np.argmax(inputModel.predict(scaledData), axis=-1)[0]
        genreList[prediction] = genreList[prediction] + 1

    print(genreList)
    predictedGenre = np.argwhere(genreList == np.amax(genreList)).flatten().tolist()
    return predictedGenre


def predictGenreOld(path, inputModel, inputScaler):
    track, sr = librosa.load(path, sr=SAMPLE_RATE)

    rootMeanSquareEnergy = librosa.feature.rms(track, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
    zeroCrossingRate = librosa.feature.zero_crossing_rate(track, frame_length=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
    spectralCentroid = librosa.feature.spectral_centroid(track, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
    spectralRolloff = librosa.feature.spectral_rolloff(track, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
    spectralFlux = librosa.onset.onset_strength(track, sr=SAMPLE_RATE)
    spectralBandwith = librosa.feature.spectral_bandwidth(track, sr=SAMPLE_RATE, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
    spectralContrast = librosa.feature.spectral_contrast(track, sr=SAMPLE_RATE, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
    spectralFlatness = librosa.feature.spectral_flatness(track, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH)[0]
    MFCC = librosa.feature.mfcc(track, n_fft=FRAME_SIZE, hop_length=HOP_LENGTH, n_mfcc=20)

    trackData = {
        "Root Mean Square Energy": [np.mean(rootMeanSquareEnergy)],
        "Zero-Crossing Rate": [np.mean(zeroCrossingRate)],
        "Spectral Centroid": [np.mean(spectralCentroid)],
        "Spectral Rolloff": [np.mean(spectralRolloff)],
        "Spectral Flux": [np.mean(spectralFlux)],
        "Spectral Bandwith": [np.mean(spectralBandwith)],
        "Spectral Contrast": [np.mean(spectralContrast)],
        "Spectral Flatness": [np.mean(spectralFlatness)],
        "MFCC1": [np.mean(MFCC[0])], "MFCC2": [np.mean(MFCC[1])], "MFCC3": [np.mean(MFCC[2])],
        "MFCC4": [np.mean(MFCC[3])], "MFCC5": [np.mean(MFCC[4])], "MFCC6": [np.mean(MFCC[5])],
        "MFCC7": [np.mean(MFCC[6])], "MFCC8": [np.mean(MFCC[7])], "MFCC9": [np.mean(MFCC[8])],
        "MFCC10": [np.mean(MFCC[9])], "MFCC11": [np.mean(MFCC[10])], "MFCC12": [np.mean(MFCC[11])],
        "MFCC13": [np.mean(MFCC[12])], "MFCC14": [np.mean(MFCC[13])], "MFCC15": [np.mean(MFCC[14])],
        "MFCC16": [np.mean(MFCC[15])], "MFCC17": [np.mean(MFCC[16])], "MFCC18": [np.mean(MFCC[17])],
        "MFCC19": [np.mean(MFCC[18])], "MFCC20": [np.mean(MFCC[19])]
    }
    df = pd.DataFrame(trackData)

    scaledData = inputScaler.transform(df.values)
    #print(inputModel.predict(scaledData))
    return np.argmax(inputModel.predict(scaledData), axis=-1)[0]