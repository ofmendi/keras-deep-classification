import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import Imputer


def main():
    df = pd.read_csv("breast-cancer-wisconsin.data")
    df.replace('?', -99999, inplace=True)
    ndf = df.drop(['1000025'], axis=1)

    imp = Imputer(missing_values=-99999, strategy="mean", axis=0)
    ndf = imp.fit_transform(ndf)

    inputs = ndf[:, 0:8]
    output = ndf[:, 9]

    model = Sequential()

    model.add(Dense(128, input_dim=8, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
 
    model.fit(inputs, output, epochs=50, batch_size=32, validation_split=0.13)

    new_data = np.array([3,1,6,1,7,2,9,4]).reshape(1, 8)
    print(model.predict_classes(new_data))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n[.] Bye... Bye..")
