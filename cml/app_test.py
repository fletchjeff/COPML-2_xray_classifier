from flask import Flask,send_from_directory,request,send_file, jsonify
import logging, os, glob, random
from IPython.display import Javascript,HTML
#from pandas.io.json import dumps as jsonify

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app= Flask(__name__,static_url_path='')
    
@app.route('/')
def index():
  return "<script> window.location.href = '/web_app/index.html'</script>"

@app.route("/random_image")
def random_image():
  normal_file = glob.glob("chest_xray/normal/*.jpeg")
  pneumonia_file = glob.glob("chest_xray/pneumonia/*.jpeg")
  all_files = normal_file + pneumonia_file
  return jsonify({'file':random.choice(all_files)})

@app.route('/web_app/<path:path>')
def send_file(path):
  return send_from_directory('web_app', path)

@app.route('/chest_xray/<path:path>')
def send_image(path):
  return send_from_directory('chest_xray', path)
  
if __name__=="__main__":
  app.run(host="127.0.0.1", 
          port=int(os.environ['CDSW_APP_PORT']))