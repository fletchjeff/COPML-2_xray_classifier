!pip3 install git+https://github.com/fastforwardlabs/cmlbootstrap#egg=cmlbootstrap

#-r requirements.txt --progress-bar off

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

# Set the setup variables needed by CMLBootstrap
HOST = os.getenv("CDSW_API_URL").split(
    ":")[0] + "://" + os.getenv("CDSW_DOMAIN")
USERNAME = os.getenv("CDSW_PROJECT_URL").split(
    "/")[6]  # args.username  # "vdibia"
API_KEY = os.getenv("CDSW_API_KEY") 
PROJECT_NAME = os.getenv("CDSW_PROJECT")  

# Instantiate API Wrapper
cml = CMLBootstrap(HOST, USERNAME, API_KEY, PROJECT_NAME)

# Set the STORAGE environment variable
try : 
  storage=os.environ["STORAGE"]
except:
  if os.path.exists("/etc/hadoop/conf/hive-site.xml"):
    tree = ET.parse('/etc/hadoop/conf/hive-site.xml')
    root = tree.getroot()
    for prop in root.findall('property'):
      if prop.find('name').text == "hive.metastore.warehouse.dir":
        storage = prop.find('value').text.split("/")[0] + "//" + prop.find('value').text.split("/")[2]
  else:
    storage = "/user/" + os.getenv("HADOOP_USER_NAME")
  storage_environment_params = {"STORAGE":storage}
  storage_environment = cml.create_environment_variable(storage_environment_params)
  os.environ["STORAGE"] = storage
  
!hdfs dfs -mkdir -p $STORAGE/user/$HADOOP_USER_NAME/data/xray/train
!hdfs dfs -mkdir -p $STORAGE/user/$HADOOP_USER_NAME/data/xray/test
!hdfs dfs -mkdir -p $STORAGE/user/$HADOOP_USER_NAME/data/xray/models
!hdfs dfs -copyFromLocal -p -f data/train $STORAGE/user/$HADOOP_USER_NAME/data/xray/train
!hdfs dfs -copyFromLocal -p -f data/test $STORAGE/user/$HADOOP_USER_NAME/data/xray/test
