from flask import Flask,send_from_directory,request,send_file, jsonify
import logging, os, glob, random
from IPython.display import Javascript,HTML
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

@app.route('/app/<path:path>')
def send_file(path):
  return send_from_directory('app', path)

@app.route('/data/<path:path>')
def send_image(path):
  return send_from_directory('data', path)
  
if __name__=="__main__":
  app.run(host="127.0.0.1", 
          port=int(os.environ['CDSW_APP_PORT']))