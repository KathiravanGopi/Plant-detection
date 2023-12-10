#app.py
from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
from flask import Flask, render_template,request,redirect,url_for
from keras.preprocessing.image import load_img
from keras.models import load_model
import numpy as np
from PIL import ImageOps,Image
 
app = Flask(__name__)
model="/home/kathiravan/Documents/keras_model.h5"
class_names = open("/home/kathiravan/Documents/labels.txt", "r").readlines()
UPLOAD_FOLDER = 'app/static/uploads/'
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif','webp'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
     
 
@app.route('/')
def home():
    return render_template('index.html')



@app.route('/', methods=['POST'])

def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        image_path = "/home/kathiravan/githubclones/YouTube-I-mostly-use-colab-now-/Flask Machine Learning Model on Heroku/Plant detection/app/static/uploads/" + file.filename
        filename = secure_filename(file.filename)
        file.save(image_path)
        np.set_printoptions(suppress=True)
        model = load_model("/home/kathiravan/Documents/keras_model.h5", compile=False)
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        image = load_img(image_path, target_size=(224, 224))
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

        # Load the image into the array
        data[0] = normalized_image_array

        # Predicts the model
        prediction = model.predict(data)
        index = np.argmax(prediction)
        kathir=class_names[index]
        confidence_score = prediction[0][index]
        classification = kathir[2:]
        
        return render_template('index.html', prediction=classification,filename=filename)
        
        

    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
 
@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)
 
if __name__ == "__main__":
    app.run(port=8000,debug=True)