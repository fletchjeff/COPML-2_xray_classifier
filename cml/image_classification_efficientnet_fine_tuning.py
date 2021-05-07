!ln -s /usr/local/cuda-11.1/targets/x86_64-linux/lib/libcusolver.so.11.0.1.105 /usr/local/cuda-11.1/targets/x86_64-linux/lib/libcusolver.so.10

import pathlib
import tensorflow as tf

tf.test.gpu_device_name()

data_dir = pathlib.Path("/home/cdsw/chest_xray")

batch_size = 32
IMG_SIZE = 224

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(IMG_SIZE, IMG_SIZE),
  batch_size=batch_size)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(IMG_SIZE, IMG_SIZE),
  batch_size=batch_size)

val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

img_augmentation = Sequential(
    [
        preprocessing.RandomRotation(factor=0.1),
        preprocessing.RandomFlip("horizontal"),
        preprocessing.RandomZoom(0.1),
    ],
    name="img_augmentation",
)  


AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)


import matplotlib.pyplot as plt

def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.ylim([0,1])
    plt.show()

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras.applications import EfficientNetB0

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

x = img_augmentation(inputs)

model = EfficientNetB0(include_top=False, input_tensor=x, weights="imagenet")
x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
x = layers.BatchNormalization(name="last_batch_norm")(x)

top_dropout_rate = 0.2
x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
outputs = layers.Dense(1, name="pred")(x)

model = tf.keras.Model(inputs, outputs, name="EfficientNet")
len(model.trainable_variables)


model.trainable = True

for i, layer in enumerate(model.layers):
  #print("=====")
  #print(layer)
  #print(i)
  if i > len(model.layers) - 21:
      #print("top layer")
      #print(layer.name)
      if not isinstance(layer, layers.BatchNormalization):
          #print("Not a Batch Layer")
          layer.trainable = True
      else:
          #print("Is Batch Layer")
          if layer.name == "last_batch_norm":
              layer.trainable = True
          else:
              layer.trainable = False
  else:
      layer.trainable = False
  #print(layer.trainable)

    
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
)


len(model.trainable_variables)


epochs = 30  
hist = model.fit(train_dataset, epochs=epochs, validation_data=validation_dataset, verbose=2)

loss, accuracy = model.evaluate(test_dataset)
print('Test accuracy :', accuracy)

plot_hist(hist)
