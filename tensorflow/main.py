# coding: utf-8

# import numpy as np
import os
import sys
import time
# import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import SGD
sys.path.append(os.pardir)

from dataset.mnist import load_mnist


def main():
    print("load mnist dataset...")
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    print("construct a model...")
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(784, )),
        keras.layers.Dense(50, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    opt = SGD(lr=0.1)
    model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    batch_size = 100
    epochs = 17
    print("start training...")
    st = time.time()
    model.fit(x_train, t_train, batch_size, epochs=epochs, verbose=False)
    print("finished.")
    train_loss, train_acc = model.evaluate(x_train, t_train)
    test_loss, test_acc = model.evaluate(x_test, t_test)
    print("loss and acc for train: {}, {}".format(train_loss, train_acc))
    print("loss and acc for test: {}, {}".format(test_loss, test_acc))
    ft = time.time()
    
    print(f"Elapsed time to train: {ft - st:.2f}.")


if __name__ == "__main__":
    main()
