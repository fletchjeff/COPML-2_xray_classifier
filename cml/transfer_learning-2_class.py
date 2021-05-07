import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing


tf.test.gpu_device_name()


AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)


# ### Use data augmentation

data_augmentation = tf.keras.Sequential([
  preprocessing.RandomFlip("horizontal"),#,input_shape=(img_height,img_width,3)),
  preprocessing.RandomRotation(0.1),
  preprocessing.RandomZoom(0.1),
])


# # Build the model

IMG_SHAPE = (224,224) + (3,)

base_model = tf.keras.applications.MobileNetV2(
  input_shape=IMG_SHAPE,
  include_top=False,
  weights='imagenet'
)

base_model.trainable = False

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

prediction_layer = tf.keras.layers.Dense(1)

inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)


# # Compile the model
base_learning_rate = 0.0001
model.compile(
  optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
  metrics=['accuracy']
)

# ### Train the model

initial_epochs = 10

history = model.fit(
  train_dataset,
  epochs=initial_epochs,
  validation_data=validation_dataset,
  verbose=2
)


# ### Accuracy Values

print(history.history['accuracy'])
print(val_acc = history.history['val_accuracy'])

print(history.history['loss'])
print(history.history['val_loss'])

# ### Fine-Tuning

base_model.trainable = True

# Fine-tune from this layer onwards
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable =  False


# ### Compile the model with new learning rate

model.compile(
  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
  optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),
  metrics=['accuracy']
)

fine_tune_epochs = 10
total_epochs =  initial_epochs + fine_tune_epochs

history_fine = model.fit(
  train_dataset,
  epochs=total_epochs,
  initial_epoch=history.epoch[-1],
  validation_data=validation_dataset
)

acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']

# ### Evaluation and prediction
loss, accuracy = model.evaluate(test_dataset)
print('Test accuracy :', accuracy)

model.save("2_class")


