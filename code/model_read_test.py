from smart_open import open


import os, boto3, requests
session = boto3.Session(
     aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'],
     aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'],
)
url = 's3://smart-open-py37-benchmark-results/test.txt'
with open(url, 'wb', transport_params={'client': session.client('s3')}) as fout:
     bytes_written = fout.write(b'hello world!')
   print(bytes_written)

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

client.upload_file('models/model_1.h5', 'demo-aws-2', 'user/jfletcher/data/xray/models/model_1.h5')

!hdfs dfs -mkdir -p $STORAGE/user/$HADOOP_USER_NAME/data/xray/models


model = tf.keras.models.load_model(client.get_object(Bucket='demo-aws-2', Key='user/jfletcher/data/xray/models/model_1.h5')['Body'].read())



session = boto3.Session(
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    aws_session_token=SESSION_TOKEN,
)
client = session.client('s3', endpoint_url=..., config=...)
fin = open('s3://bucket/key', transport_params=dict(client=client))