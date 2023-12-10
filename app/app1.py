from flask import Flask, render_template,request,redirect,url_for
from keras.preprocessing.image import load_img
from keras.models import load_model
import numpy as np
from PIL import ImageOps,Image
from werkzeug.utils import secure_filename
import urllib.request

app=Flask(__name__)
model="/home/kathiravan/Documents/keras_model.h5"
class_names = open("/home/kathiravan/Documents/labels.txt", "r").readlines()

@app.route('/',methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route("/",methods=['POST'])
def predict():
     imagefile= request.files['imagefile']
     image_path = "/home/kathiravan/githubclones/YouTube-I-mostly-use-colab-now-/Flask Machine Learning Model on Heroku/Part 2/images/" + imagefile.filename
     imagefile.save(image_path)
     
     # Disable scientific notation for clarity
     np.set_printoptions(suppress=True)

     # Load the model
     model = load_model("/home/kathiravan/Documents/keras_model.h5", compile=False)

# Load the labels
     class_names = open("/home/kathiravan/Documents/labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
     data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
     image = load_img(image_path, target_size=(224, 224))
    
    # resizing the image to be at least 224x224 and then cropping from the center
     size = (224, 224)
     image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
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
     
     return render_template('index.html', prediction=classification)


if __name__ =='__main__':  
    app.run(port=8000,debug=True)