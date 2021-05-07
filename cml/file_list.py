

import numpy as np
from sklearn.model_selection import train_test_split
import glob
import shutil

all_normal_files = glob.glob("/home/cdsw/dataset/all/normal/*.jpeg")
all_virus_files = glob.glob("/home/cdsw/dataset/all/virus/*.jpeg")
all_bacteria_files = glob.glob("/home/cdsw/dataset/all/bacteria/*.jpeg")
train_normal_files, test_normal_files = train_test_split(
  all_normal_files, test_size=0.2, random_state=42)

train_virus_files, test_virus_files = train_test_split(
  all_virus_files, test_size=0.2, random_state=42)

train_bacteria_files, test_bacteria_files = train_test_split(
  all_bacteria_files, test_size=0.2, random_state=42)

for filename in test_normal_files:
  shutil.move(filename,filename.replace("all","test"))
  
for filename in test_virus_files:
  shutil.move(filename,filename.replace("all","test"))

for filename in test_bacteria_files:
  shutil.move(filename,filename.replace("all","test"))  

for filename in train_normal_files:
  shutil.move(filename,filename.replace("all","train"))
  
for filename in train_virus_files:
  shutil.move(filename,filename.replace("all","train"))
  
for filename in train_ba_files:
  shutil.move(filename,filename.replace("all","train"))

#import tensorflow as tf
#import pathlib
#data_dir = pathlib.Path("/home/cdsw/dataset/all")
#
#batch_size = 32
#img_height = 600 #600
#img_width = 600 #600
##IMG_SIZE = (456, 456)
#
#train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
#  data_dir,
#  validation_split=0.2,
#  subset="training",
#  seed=123,
#  image_size=(img_height, img_width),
#  batch_size=batch_size)
#
#validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
#  data_dir,
#  validation_split=0.2,
#  subset="validation",
#  seed=123,
#  image_size=(img_height, img_width),
#  batch_size=batch_size)
#
#val_batches = tf.data.experimental.cardinality(validation_dataset)
#test_dataset = validation_dataset.take(val_batches // 5)
#validation_dataset = validation_dataset.skip(val_batches // 5)