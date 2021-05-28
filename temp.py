import os
import sys

from PIL import Image
from io import BytesIO
import os
import requests
import numpy as np
import boto3
#from tensorflow.keras.layers.experimental import preprocessing
import matplotlib.pyplot as plt
from cmlbootstrap import CMLBootstrap
from subprocess import check_output
from smart_open import open



# # S3 connection
# This will establish a boto3 connection to the S3 bucket where the images are stored

cml = CMLBootstrap()

client = cml.boto3_client(os.environ['IDBROKER'])

# # Fetch the test and training data

def get_file_list(path):
  file_list = []
  hdfs_outout = check_output(['hdfs', 'dfs', '-ls' , path],universal_newlines=True)
  for file in hdfs_outout.split("\n")[1:-1]:
    if file.split(" ")[0][0] == '-':
      file_list.append(file.split(" ")[-1])#.split("/")[-1])
  return file_list

image_storage = "{}/user/{}/data/xray".format(os.environ['STORAGE'],os.environ["HADOOP_USER_NAME"])

batch_size = 32
img_size = 224 

train_file_list = get_file_list(image_storage + '/train/normal')
train_file_list = train_file_list + get_file_list(image_storage + '/train/bacteria')
train_file_list = train_file_list + get_file_list(image_storage + '/train/virus')