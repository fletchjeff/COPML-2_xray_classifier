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

# # S3 connection
# This will establish a boto3 connection to the S3 bucket where the images are stored

IDBROKER_URL = "demo-aws-2-dl-idbroker0.demo-aws.ylcu-atmi.cloudera.site"

from requests_kerberos import HTTPKerberosAuth
r = requests.get("https://{}:8444/gateway/dt/knoxtoken/api/v1/token".format(IDBROKER_URL), auth=HTTPKerberosAuth())

url = "https://{}:8444/gateway/aws-cab/cab/api/v1/credentials".format(IDBROKER_URL)
headers = {
    'Authorization': "Bearer "+ r.json()['access_token'],
    'cache-control': "no-cache"
    }

response = requests.request("GET", url, headers=headers)

ACCESS_KEY=response.json()['Credentials']['AccessKeyId']
SECRET_KEY=response.json()['Credentials']['SecretAccessKey']
SESSION_TOKEN=response.json()['Credentials']['SessionToken']


client = boto3.client(
    's3',
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    aws_session_token=SESSION_TOKEN,
)

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
img_size = 224 

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

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

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

fine_tune_epochs = 30
total_epochs =  initial_epochs + fine_tune_epochs

history_fine = model.fit(
  train_dataset,
  epochs=total_epochs,
  initial_epoch=history.epoch[-1],
  validation_data=validation_dataset,
  verbose=2
)

plt.plot(history_fine.history['accuracy'])
plt.plot(history_fine.history['val_accuracy'])

plt.plot(history_fine.history['loss'])
plt.plot(history_fine.history['val_loss'])

# ### Evaluation and prediction
loss, accuracy = model.evaluate(test_dataset)
print('Test accuracy :', accuracy)

model.save("model_2.h5")
