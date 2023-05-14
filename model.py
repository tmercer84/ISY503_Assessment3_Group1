"""

    This code is adapted from https://github.com/naokishibuya/car-behavioral-cloning by Naoki Shibuya, Darío Hereñú, Ferdinand Mütsch, Abhijeet Singh.
    The original code can be found at: https://github.com/naokishibuya/car-behavioral-cloning
    
    This code was addapted by:
        Eliana Trujillo - A00064723
        Mariana Laranjeira - A00027796
        Thais Prata - A00083807
        Thiago Mercer - A00059953
    
    For the subject ISY503 - Intelligent Systems - Assessment 3
    
"""

# This is the imports used in the code
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, Dropout, Dense, Flatten
from utils import INPUT_SHAPE, batch_generator
import argparse
import os

np.random.seed(0)


def load_data(args):
    """
    Load training data and split it into training and validation set
    """
    data_df = pd.read_csv(os.path.join(args.data_dir, 'driving_log.csv'))

    X = data_df[['center', 'left', 'right']].values
    y = data_df['steering'].values

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.test_size, random_state=0)

    return X_train, X_valid, y_train, y_valid


def build_model(args):
    """
    Modified NVIDIA model
    """
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
    model.add(Conv2D(24, (5, 5), activation='elu', strides=(1, 1), padding='valid'))
    model.add(Conv2D(36, (5, 5), activation='elu', strides=(1, 1), padding='valid'))
    model.add(Conv2D(48, (5, 5), activation='elu', strides=(1, 1), padding='valid'))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Dropout(args.keep_prob))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()

    return model


def train_model(model, args, X_train, X_valid, y_train, y_valid):
    """
    Train the model
    """
    checkpoint = ModelCheckpoint('model.h5', 
                                 monitor='val_loss', 
                                 save_best_only=True)


    model.compile(loss='mean_squared_error', 
                  optimizer=Adam(lr=args.learning_rate))

    model.fit_generator(
        batch_generator(args.data_dir, 
                                        X_train, 
                                        y_train, 
                                        args.batch_size, 
                                        True),
        steps_per_epoch=len(X_train) // args.batch_size,
        epochs=args.nb_epoch,
        validation_data=batch_generator(args.data_dir, 
                                        X_valid, 
                                        y_valid, 
                                        args.batch_size, 
                                        False),
        validation_steps=len(X_valid) // args.batch_size,
        callbacks=[checkpoint],
        verbose=1)


def s2b(s):
    """
    Converts a string to boolean value
    """
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'


def main():
    """
    Load train/validation data set and train the model
    """
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default='data')
    parser.add_argument('-t', help='test size fraction',    dest='test_size',         type=float, default=0.2)
    parser.add_argument('-k', help='drop out probability',  dest='keep_prob',         type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs',      dest='nb_epoch',          type=int,   default=10)
    parser.add_argument('-s', help='samples per epoch',     dest='samples_per_epoch', type=int,   default=20000)
    parser.add_argument('-b', help='batch size',            dest='batch_size',        type=int,   default=40)
    parser.add_argument('-o', help='save best models only', dest='save_best_only',    type=s2b,   default='true')
    parser.add_argument('-l', help='learning rate',         dest='learning_rate',     type=float, default=1.0e-4)
    args = parser.parse_args()

    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    data = load_data(args)
    model = build_model(args)
    train_model(model, args, *data)


if __name__ == '__main__':
    main()

