!pip3 install --upgrade git+https://github.com/fletchjeff/cmlbootstrap#egg=cmlbootstrap
!pip3 install -r requirements.txt --progress-bar off

# Create the directories and upload data

from cmlbootstrap import CMLBootstrap
import os
from PIL import Image
import xml.etree.ElementTree as ET
from smart_open import open
from subprocess import check_output
import glob
import logging
logging.basicConfig(level=logging.WARN)

# Instantiate API Wrapper
cml = CMLBootstrap()
client = cml.boto3_client(cml.get_id_broker())

# Set the STORAGE environment variable
try : 
  storage=os.environ["STORAGE"]
except:
  try:
    storage = cml.get_cloud_storage()
    storage_environment_params = {"STORAGE":storage}
    storage_environment = cml.create_environment_variable(storage_environment_params)
    os.environ["STORAGE"] = storage
  except:
    storage = "/user/" + os.environ["HADOOP_USER_NAME"]
    
# Set the IDBROKER environment variable
try : 
  id_broker=os.environ["IDBROKER"]
except:
  try:
    id_broker = cml.get_id_broker()
    id_broker_environment_params = {"IDBROKER":id_broker}
    id_brokere_environment = cml.create_environment_variable(id_broker_environment_params)
    os.environ["IDBROKER"] = id_broker
  except:
    print("ID Broker hostname not avabilable")
    
image_storage = "{}/user/{}/data/xray/".format(storage,os.environ["HADOOP_USER_NAME"])

image_storage = image_storage + "{}/{}/{}"

for directory in glob.glob("data/train/*"):
  directory = directory.split("/")[-1]
  print("====")
  print("Processing train/{} directory".format(directory))
  for i,image in enumerate(glob.glob("data/train/{}/*".format(directory))):
    pil_image = Image.open(image)
    write_url  = image_storage.format('train',directory,image.split("/")[-1])
    smart_open_writer = open(write_url, 'wb', transport_params={'client': client})
    pil_image.save(smart_open_writer)
    smart_open_writer.close()
    if i%50 == 0:
      print("Uploaded {} images".format(i))
      print("Last image uploaded: {}".format(image))
      
for directory in glob.glob("data/test/*"):
  directory = directory.split("/")[-1]
  print("====")
  print("Processing test/{} directory".format(directory))
  for i,image in enumerate(glob.glob("data/test/{}/*".format(directory))):
    pil_image = Image.open(image)
    write_url  = image_storage.format('test',directory,image.split("/")[-1])
    smart_open_writer = open(write_url, 'wb', transport_params={'client': client})
    pil_image.save(smart_open_writer)
    smart_open_writer.close()
    if i%50 == 0:
      print("Uploaded {} images".format(i))
      print("Last image uploaded: {}".format(image))     

!hdfs dfs -ls -R $STORAGE/user/$HADOOP_USER_NAME/data/xray/ | wc -l

#!hdfs dfs -mkdir -p $STORAGE/user/$HADOOP_USER_NAME/data/xray/train
#!hdfs dfs -mkdir -p $STORAGE/user/$HADOOP_USER_NAME/data/xray/test
#!hdfs dfs -mkdir -p $STORAGE/user/$HADOOP_USER_NAME/data/xray/models
#!hdfs dfs -copyFromLocal -p -f data/train $STORAGE/user/$HADOOP_USER_NAME/data/xray/train
#!hdfs dfs -copyFromLocal -p -f data/test $STORAGE/user/$HADOOP_USER_NAME/data/xray/test
