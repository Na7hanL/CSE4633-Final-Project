import numpy as np
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


if __name__ == '__main__':
    data_dir = './CUB_200_2011/CUB_200_2011/images/'
    data_dir = pathlib.Path(data_dir).with_suffix('')


    image_count = len(list(data_dir.glob('*/*.jpg')))
    print(f"\nTotal Number of Images: {image_count}")

    batch_size = 32
    img_height = 180
    img_width = 180

    print('\nCollecting Training Data...')
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    
    print('\nCollecting Validation Data...')

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    
    class_names = train_ds.class_names
    
    plt.figure(figsize=(10, 10))

    print(train_ds.take(1))

    for images, labels in train_ds.take(1):
        print(images)
        #print(images)for i in range(9):
        #print(images)  ax = plt.subplot(3, 3, i + 1)
        #print(images)  plt.imshow(images[i].numpy().astype("uint8"))
        #print(images)  plt.title(class_names[labels[i]])
        #  plt.axis("off")
    
    