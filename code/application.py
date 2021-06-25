from flask import Flask,send_from_directory,request,send_file, jsonify
import logging, os, glob, random
import numpy as np
import tensorflow as tf
from random import sample
from lime import lime_image
from PIL import Image
from skimage.segmentation import mark_boundaries
from io import BytesIO  
import base64
from cmlbootstrap import CMLBootstrap
#from pandas.io.json import dumps as jsonify

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app= Flask(__name__,static_url_path='')
    
@app.route('/')
def index():
  return "<script> window.location.href = '/app/index.html'</script>"

@app.route("/random_image")
def random_image():
  normal_file = glob.glob("data/test/normal/*.jpeg")
  bacteria_file = glob.glob("data/test/bacteria/*.jpeg")
  virus_file = glob.glob("data/test/virus/*.jpeg")
  all_files = normal_file + bacteria_file + virus_file
  return jsonify({'file':random.choice(all_files)})

@app.route("/explain_image", methods=['GET'])
def explain_image():
  model_1 = tf.keras.models.load_model('/home/cdsw/models/model_1.h5')
  sample_image = request.args.get('image', '')
  im = Image.open(sample_image)
  im = im.resize((224,224),Image.ANTIALIAS)
  im = im.convert("RGB")
  im_array = tf.keras.preprocessing.image.img_to_array(im)
  im_array = np.expand_dims(im_array, axis=0)

  # Train explainer
  explainer = lime_image.LimeImageExplainer()
  explanation = explainer.explain_instance(im_array[0].astype('double'), model_1.predict)

  # Create boudries
  temp_1, mask_1 = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
  blank = Image.new('RGB', (224, 224))
  explained_image = mark_boundaries(blank,mask_1,mode='thick',color=(0,0.75,1))
  img = Image.fromarray((explained_image * 255).astype(np.uint8))
  img = img.convert("RGBA")

  # Set Alpha
  datas = img.getdata()
  newData = []
  for item in datas:
      if item[0] == 0 and item[1] == 0 and item[2] == 0:
          newData.append((0, 0, 0, 0))
      else:
          newData.append(item)

  img.putdata(newData)

  # Encode to base64
  output = BytesIO()
  #img_read = Image.open(sample_image)
  img.save(output, format='PNG')
  im_data = output.getvalue()
  image_data = base64.b64encode(im_data)
  if not isinstance(image_data, str):
      # Python 3, decode from bytes to string
      image_data = image_data.decode()
  data_url = 'data:image/png;base64,' + image_data
  return jsonify({'image':data_url})

@app.route("/model_access_keys")
def model_access_keys():
  cml = CMLBootstrap()
  project_id = cml.get_project()['id']
  params = {"projectId":project_id,"latestModelDeployment":True,"latestModelBuild":True}
  for model in cml.get_models(params):
    if model['name'] == 'XRay Model 1':
      model_id_1 = model['id']
    else:
      model_id_2 = model['id']

  latest_model_1 = cml.get_model({"id": model_id_1, "latestModelDeployment": True, "latestModelBuild": True})
  latest_model_2 = cml.get_model({"id": model_id_2, "latestModelDeployment": True, "latestModelBuild": True})
  return jsonify({
    'model_1_access_key': latest_model_1['accessKey'],
    'model_2_access_key': latest_model_2['accessKey']
  })


@app.route('/app/<path:path>')
def send_file(path):
  return send_from_directory('app', path)

@app.route('/data/<path:path>')
def send_image(path):
  return send_from_directory('data', path)
  
if __name__=="__main__":
  app.run(host="127.0.0.1", 
          port=int(os.environ['CDSW_APP_PORT']))