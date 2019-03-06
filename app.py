#Usage: python app.py
import os

import sys
sys.path.insert(0, os.path.join(os.getcwd(), "model"))
import lpr_model

from flask import Flask, render_template, request, redirect, url_for
from werkzeug import secure_filename
import numpy as np
import imutils
import cv2
import time
import uuid
import base64

lpr = lpr_model.EAST_CRNN()
lpr.load(east_path="model/trained_model_weights/frozen_east_text_detection.pb",
         crnn_path="model/trained_model_weights/CRNN30.pth")

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png', 'JPG'])

def get_as_base64(url):
    return base64.b64encode(requests.get(url).content)

def my_random_string(string_length=10):
    """Returns a random string of length string_length."""
    random = str(uuid.uuid4()) # Convert UUID format to a Python string.
    random = random.upper() # Make all characters uppercase.
    random = random.replace("-","") # Remove the UUID '-'.
    return random[0:string_length] # Return the random string.

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def template_test():
    return render_template('template.html', imagesource='../uploads/intro.jpg')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        
        file = request.files['file']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            start_time = time.time()
            result = lpr.predict(file_path)
            end_time = time.time()

            # lpr.cropped_image()
            
            filename = my_random_string(6) + filename
            os.rename(file_path, os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
            return render_template('template.html', label=result, imagesource='../uploads/' + filename,
                                    P1=result[0][0], P2=result[1][0], P3=result[2][0], P4=result[3][0], P5=result[4][0],
                                    C1=np.round(result[0][1]*100, 2), C2=np.round(result[1][1]*100, 2), 
                                    C3=np.round(result[2][1]*100, 2), C4=np.round(result[3][1]*100, 2), 
                                    C5=np.round(result[4][1]*100, 2), time=np.round(end_time-start_time, 2))

from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

from werkzeug import SharedDataMiddleware
app.add_url_rule('/uploads/<filename>', 'uploaded_file',
                 build_only=True)
app.wsgi_app = SharedDataMiddleware(app.wsgi_app, {
    '/uploads':  app.config['UPLOAD_FOLDER']
})

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=False)