!pip3 install --upgrade git+https://github.com/fletchjeff/cmlbootstrap#egg=cmlbootstrap
!pip3 install -r requirements.txt --progress-bar off

# Create the directories and upload data

from cmlbootstrap import CMLBootstrap
from IPython.display import Javascript, HTML
import os
import time
import json
import requests
import xml.etree.ElementTree as ET
import datetime

run_time_suffix = datetime.datetime.now()
run_time_suffix = run_time_suffix.strftime("%d%m%Y%H%M%S")

# Instantiate API Wrapper
cml = CMLBootstrap()

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
    storage = "/user/" + os.getenv("HADOOP_USER_NAME")
    
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

!hdfs dfs -mkdir -p $STORAGE/user/$HADOOP_USER_NAME/data/xray/train
!hdfs dfs -mkdir -p $STORAGE/user/$HADOOP_USER_NAME/data/xray/test
!hdfs dfs -mkdir -p $STORAGE/user/$HADOOP_USER_NAME/data/xray/models
!hdfs dfs -copyFromLocal -p -f data/train $STORAGE/user/$HADOOP_USER_NAME/data/xray/train
!hdfs dfs -copyFromLocal -p -f data/test $STORAGE/user/$HADOOP_USER_NAME/data/xray/test
