!ln -s /usr/local/cuda-11.1/targets/x86_64-linux/lib/libcusolver.so.11.0.1.105 /usr/local/cuda-11.1/targets/x86_64-linux/lib/libcusolver.so.10

from PIL import Image
from io import BytesIO
import os
import requests
import tensorflow as tf
import numpy as np
import boto3
from tensorflow.keras.layers.experimental import preprocessing
import matplotlib.pyplot as plt
from cmlbootstrap import CMLBootstrap

# # S3 connection
# This will establish a boto3 connection to the S3 bucket where the images are stored

cml = CMLBootstrap()

client = cml.boto3_client(os.environ['IDBROKER'])
# # Fetch the test and training data

def get_file_paths(prefix):
  keys = []
  kwargs = {'Bucket': s3_bucket, 'Prefix' : prefix }
  while True:
      resp = client.list_objects_v2(**kwargs)
      for obj in resp['Contents']:
          keys.append(obj['Key'])
      try:
          kwargs['ContinuationToken'] = resp['NextContinuationToken']
      except KeyError:
          break    
  return keys

s3_bucket = os.getenv("STORAGE").split("//")[1]

batch_size = 32
img_size = 456
IMG_SIZE = 456

#train_file_list = get_file_paths('datalake/data/xray/train/normal')
train_file_list = get_file_paths('datalake/data/xray/train/bacteria')
train_file_list = train_file_list + get_file_paths('datalake/data/xray/train/virus')

image_data_array = np.empty(shape=(len(train_file_list),img_size, img_size, 3),dtype='int32')
label_data_array = np.empty(shape=(len(train_file_list),),dtype='int32')
            
for i,path in enumerate(train_file_list):
  image_raw = client.get_object(Bucket=s3_bucket, Key=path)['Body'].read()       
  img = Image.open(BytesIO(image_raw))
  img = img.resize((img_size,img_size),Image.ANTIALIAS)
  img = img.convert("RGB")
  img_np_array = np.asarray(img)
  image_data_array[i] = img_np_array
  if 'bacteria' in path:
    label_data_array[i] = 0
  else:
    label_data_array[i] = 1
  if i%50 == 0:
    print("loading training file number: {}/{} with path: {}".format(i+1,len(train_file_list),path))

train_dataset = tf.data.Dataset.from_tensor_slices((image_data_array, label_data_array))
train_dataset = train_dataset.shuffle(len(train_dataset), seed=123, reshuffle_each_iteration=False)
train_dataset.colnames = ['bacteria','virus']
train_dataset = train_dataset.batch(batch_size)

train_batches = tf.data.experimental.cardinality(train_dataset)
validation_dataset = train_dataset.take(train_batches // 5)
train_dataset = train_dataset.skip(train_batches // 5)

#test_file_list = get_file_paths('datalake/data/xray/test/normal')
test_file_list =get_file_paths('datalake/data/xray/test/bacteria')
test_file_list = test_file_list + get_file_paths('datalake/data/xray/test/virus')

image_data_array = np.empty(shape=(len(test_file_list),img_size, img_size, 3),dtype='int32')
label_data_array = np.empty(shape=(len(test_file_list),),dtype='int32')
            
for i,path in enumerate(test_file_list):
  image_raw = client.get_object(Bucket=s3_bucket, Key=path)['Body'].read()       
  img = Image.open(BytesIO(image_raw))
  img = img.resize((img_size,img_size),Image.ANTIALIAS)
  img = img.convert("RGB")
  img_np_array = np.asarray(img)
  image_data_array[i] = img_np_array
  if 'bacteria' in path:
    label_data_array[i] = 0
  else:
    label_data_array[i] = 1
  if i%50 == 0:
    print("loading test file number: {}/{} with path: {}".format(i+1,len(test_file_list),path))

test_dataset = tf.data.Dataset.from_tensor_slices((image_data_array, label_data_array))
test_dataset = test_dataset.shuffle(len(test_dataset), seed=123, reshuffle_each_iteration=False)
test_dataset.colnames = ['bacteria','virus']
test_dataset = test_dataset.batch(batch_size)

# # Model Training

tf.test.gpu_device_name()


AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

del image_data_array
del label_data_array



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
from tensorflow.keras.applications import EfficientNetB5

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

x = img_augmentation(inputs)

model = EfficientNetB5(include_top=False, input_tensor=x, weights="imagenet")
x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
x = layers.BatchNormalization(name="last_batch_norm")(x)

top_dropout_rate = 0.2
x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
outputs = layers.Dense(1, name="pred")(x)

model = tf.keras.Model(inputs, outputs, name="EfficientNet")
len(model.trainable_variables)


model.trainable = True

for i, layer in enumerate(model.layers):
  if i > len(model.layers) - 21:
      if not isinstance(layer, layers.BatchNormalization):
          layer.trainable = True
      else:
          if layer.name == "last_batch_norm":
              layer.trainable = True
          else:
              layer.trainable = False
  else:
      layer.trainable = False
    
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

model.save("models/model_2_efficient.h5")

