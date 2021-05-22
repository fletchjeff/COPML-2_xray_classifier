from smart_open import open
from cmlbootstrap import CMLBootstrap
from PIL import Image
from io import BytesIO
from subprocess import check_output
import glob

cml = CMLBootstrap()
client = cml.boto3_client(cml.get_id_broker())


xray_storage = 's3://demo-aws-2/user/jfletcher/data/xray/smart_open/{}/{}/{}'

for directory in glob.glob("data/test/*"):
  directory = directory.split("/")[-1]
  print("====")
  print("Processing test/{} directory".format(directory))
  for i,image in enumerate(glob.glob("data/test/{}/*".format(directory))[:5]):
    print(image)
    pil_image = Image.open(image)
    write_url  = xray_storage.format('test',directory,image.split("/")[-1])
    smart_open_writer = open(write_url, 'wb', transport_params={'client': client})
    pil_image.save(smart_open_writer)
    smart_open_writer.close()
    if i%50 == 0:
      print("Uploaded {} images".format(i))


!hdfs dfs -rm -f -r $STORAGE/user/$HADOOP_USER_NAME/data/xray/smart_open/

from subprocess import Popen, PIPE
import sys
proc = Popen(['hdfs', 'dfs', '-ls' , 's3a://demo-aws-2/user/jfletcher/data/xray/smart_open/test/virus/'], stdout=PIPE, stderr=PIPE)
stdout, stderr = proc.communicate()


out = check_output(['hdfs', 'dfs', '-ls' , 's3a://demo-aws-2/user/jfletcher/data/xray/smart_open/test/virus/'],universal_newlines=True)
for lines in out.split("\n"):
    print(lines.split(" ")[-1].split("/")[-1])