import argparse
import base64
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import json


class Detector:
    def __init__(self):
        self.model = None
        self.IMG_HEIGHT = 128
        self.IMG_WIDTH = 128
        self.class_indices = {}

    def train(self, image_path, show_stats=False):
        train_dir = os.path.join(image_path, 'train')
        validation_dir = os.path.join(image_path, 'validation')

        total_train = 0
        total_val = 0

        for category in os.listdir(train_dir):
            cat_dir_train = os.path.join(train_dir, category)
            cat_dir_val = os.path.join(validation_dir, category)
            total_train += len(os.listdir(cat_dir_train))
            total_val += len(os.listdir(cat_dir_val))

        print("Total training images:", total_train)
        print("Total validation images:", total_val)

        batch_size = 32
        epochs = 10

        train_image_generator = ImageDataGenerator(rescale=1. / 255,
                                                   zoom_range=0.2,
                                                   width_shift_range=0.2,
                                                   height_shift_range=0.2)  # Generator for our training data
        validation_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our validation data

        train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                                   directory=train_dir,
                                                                   shuffle=True,
                                                                   target_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
                                                                   class_mode='sparse')

        val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                                      directory=validation_dir,
                                                                      target_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
                                                                      class_mode='sparse')

        self.class_indices = train_data_gen.class_indices
        print(self.class_indices)

        self.model = Sequential([
            Conv2D(32, 3, padding='same', activation='relu', input_shape=(self.IMG_HEIGHT, self.IMG_WIDTH, 3)),
            MaxPooling2D((2, 2), padding='same'),
            Conv2D(32, 3, padding='same', activation='relu'),
            MaxPooling2D((2, 2), padding='same'),
            Conv2D(64, 3, padding='same', activation='relu'),
            MaxPooling2D((2, 2), padding='same'),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(9, activation='sigmoid')
        ])

        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

        self.model.summary()

        history = self.model.fit_generator(
            train_data_gen,
            steps_per_epoch=total_train // batch_size,
            epochs=epochs,
            validation_data=val_data_gen,
            validation_steps=total_val // batch_size
        )
        if show_stats:
            self._show_training_stats(history, epochs)

    @staticmethod
    def _show_training_stats(history, epochs):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()

    def predict(self, image):
        input_image = np.expand_dims(image, axis=0)
        if not self.model:
            print("Please load or train a model first.")
        else:
            result = self.model.predict(input_image)
            result_dict = {}
            for k, v in self.class_indices.items():
                result_dict[k] = result[0][v]
            return result_dict

    def load_model(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        with open("{}_indices.json".format(model_path.rstrip('.h5'))) as json_file:
            self.class_indices = json.load(json_file)

    def preprocess_image_from_path(self, image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.resize(image, [self.IMG_HEIGHT, self.IMG_WIDTH], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # image = tf.image.convert_image_dtype(image, dtype=tf.float32, saturate=False)
        image = image / 255
        return image

    def preprocess_image_from_base64(self, base64_string):
        image = base64.b64decode(base64_string)
        image = tf.image.decode_png(image, channels=3)
        image = tf.image.resize(image, [self.IMG_HEIGHT, self.IMG_WIDTH], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        image = image / 255
        return image

    def store_model(self, model_path):
        if not not self.model:
            self.model.save('{}.h5'.format(model_path))
            with open("{}_indices.json".format(model_path), "w") as json_file:
                json.dump(self.class_indices, json_file)
        else:
            print("Please train the model first.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--predict", help="predict the class of a given image",  action="store")
    parser.add_argument("-t", "--train",
                        help="train the detector with the given training path that contains the images",
                        action="store")
    parser.add_argument("-m", "--model", help="path of the trained model", action="store")
    args = parser.parse_args()
    detector = Detector()
    if args.predict:
        detector.load_model(args.model)
        pi = detector.preprocess_image_from_path(args.predict)
        prediction = detector.predict(pi)
        print(prediction)
    elif args.train:
        detector.train(args.train, show_stats=True)
        detector.store_model(args.model)
