import cdsw
import pandas as pd
from sklearn.metrics import classification_report
from cmlbootstrap import CMLBootstrap
from random import sample

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

model_metrics_1 = cdsw.read_metrics(model_crn=latest_model_1["crn"],model_deployment_crn=latest_model_1["latestModelDeployment"]["crn"])
metrics_df_1 = pd.json_normalize(model_metrics_1["metrics"])
metrics_df_1 = metrics_df_1.sort_values('startTimeStampMs',ascending=True)
metrics_df_1_actual = metrics_df_1['metrics.actual_value'][metrics_df_1.shape[0]-100:-1].to_numpy()
metrics_df_1_prediction = metrics_df_1['metrics.prediction'][metrics_df_1.shape[0]-100:-1].to_numpy()

model_metrics_2 = cdsw.read_metrics(model_crn=latest_model_2["crn"],model_deployment_crn=latest_model_2["latestModelDeployment"]["crn"])
metrics_df_2 = pd.json_normalize(model_metrics_2["metrics"])
metrics_df_2 = metrics_df_2.sort_values('startTimeStampMs',ascending=True)
metrics_df_2_actual = metrics_df_2['metrics.actual_value'][metrics_df_2.shape[0]-50:-1].to_numpy()
metrics_df_2_prediction = metrics_df_2['metrics.prediction'][metrics_df_2.shape[0]-50:-1].to_numpy()


print(classification_report(metrics_df_1_actual,metrics_df_1_prediction))

print(classification_report(metrics_df_2_actual,metrics_df_2_prediction))