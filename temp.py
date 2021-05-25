params = {"projectId":1089,"latestModelDeployment":True,"latestModelBuild":True}

for job in cml.get_jobs({}):
  if job['name'] == 'Train EfficientNet':
    print(job['id'])