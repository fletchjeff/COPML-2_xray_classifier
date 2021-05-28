import cdsw
import pandas as pd
from cmlbootstrap import CMLBootstrap
import glob
from random import sample
from io import BytesIO  
import base64
from PIL import Image

## Set the model ID
# Get the model id from the model you deployed in step 5. These are unique to each 
# model on CML.

cml = CMLBootstrap()

project_id = cml.get_project()['id']
params = {"projectId":project_id,"latestModelDeployment":True,"latestModelBuild":True}

# Note about model_id
# model_id_1 = "sdfsdf"
# model_id_2 = "32sdfsdf4"

for model in cml.get_models(params):
  if model['name'] == 'XRay Model 1':
    model_id_1 = model['id']
  else:
    model_id_2 = model['id']

latest_model_1 = cml.get_model({"id": model_id_1, "latestModelDeployment": True, "latestModelBuild": True})

latest_model_2 = cml.get_model({"id": model_id_2, "latestModelDeployment": True, "latestModelBuild": True})

# Model_CRN = latest_model_1 ["crn"]
# Deployment_CRN = latest_model_1["latestModelDeployment"]["crn"]
# model_endpoint = HOST.split("//")[0] + "//modelservice." + HOST.split("//")[1] + "/model"

# Get 100 samples  
test_normal = glob.glob('/home/cdsw/data/test/normal/*')
test_pneumonia = glob.glob('/home/cdsw/data/test/virus/*') + glob.glob('/home/cdsw/data/test/bacteria/*')
sample_image_set = sample(test_normal,50) + sample(test_pneumonia,50)

# # Create an array of model responses.
# response_labels_sample = []

# Make 1000 calls to the model with increasing error
# percent_counter = 0
# percent_max = len(df_sample_clean)
for image_path in sample_image_set:
  output = BytesIO()
  img = Image.open(image_path)
  img.save(output, format='PNG')
  im_data = output.getvalue()
  image_data = base64.b64encode(im_data)
  data_url = 'data:image/png;base64,' + image_data.decode()
  data_args = {
    "path": image_path[11:-1],
    "image":data_url
  }
  print("Quering model 1 for image: " + image_path)
  cdsw.call_model(latest_model_1["accessKey"],data_args)
  if "normal" not in image_path:
    print("Quering model 2 for image: " + image_path)
    cdsw.call_model(latest_model_2["accessKey"],data_args)


# latest_model_1 = cml.get_model({"id": model_id_1, "latestModelDeployment": True, "latestModelBuild": True})

# Read in the model metrics dict.
model_metrics_1 = cdsw.read_metrics(model_crn=latest_model_1["crn"],model_deployment_crn=latest_model_1["latestModelDeployment"]["crn"])

model_metrics_2 = cdsw.read_metrics(model_crn=latest_model_2["crn"],model_deployment_crn=latest_model_2["latestModelDeployment"]["crn"])

# This is a handy way to unravel the dict into a big pandas dataframe.
metrics_df_1 = pd.json_normalize(model_metrics_1["metrics"])
metrics_df_2 = pd.json_normalize(model_metrics_2["metrics"])

def update_model_metics(row,model_number):
  prediction_uuid = row['predictionUuid']
  if model_number == 1:
    if "normal" in row['metrics.img_path']:
      actual_value = "normal"
    else:
      actual_value = "pneumonia"
    cdsw.track_delayed_metrics({"actual_value":actual_value}, prediction_uuid)
    return "Updated record {}".format(prediction_uuid)
  else:
    if "virus" in row['metrics.img_path']:
      actual_value = "virus"
    else:
      actual_value = "bacteria"
    cdsw.track_delayed_metrics({"actual_value":actual_value}, prediction_uuid)
    return "Updated record {}".format(prediction_uuid)

metrics_df_1.apply(update_model_metics,args=(1,),axis=1)
metrics_df_2.apply(update_model_metics,args=(2,),axis=1)


