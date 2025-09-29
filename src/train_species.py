"""Training script outline for species classifier.
This script expects a directory structure:
    data/species_dataset/
        species_1/
            img1.jpg
            img2.jpg
        species_2/
            ...
It does NOT automatically download datasets. Use Kaggle or your own images.
"""
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

DATA_DIR = os.path.join(os.getcwd(), 'data', 'species_dataset')
SAVE_PATH = os.path.join(os.getcwd(), 'artifacts', 'species_model.h5')

def train_species_model(img_size=(224,224), batch_size=32, epochs=10):
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Put your species image folders in {DATA_DIR}")
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2,
                                 horizontal_flip=True, rotation_range=20, zoom_range=0.2)
    train_gen = datagen.flow_from_directory(DATA_DIR, target_size=img_size, batch_size=batch_size, subset='training')
    val_gen = datagen.flow_from_directory(DATA_DIR, target_size=img_size, batch_size=batch_size, subset='validation')

    model = models.Sequential([
        layers.Input(shape=img_size + (3,)),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.MaxPool2D(),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPool2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(train_gen.num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_gen, validation_data=val_gen, epochs=epochs)
    model.save(SAVE_PATH)
    print('Saved species model to', SAVE_PATH)

if __name__ == '__main__':
    train_species_model()
