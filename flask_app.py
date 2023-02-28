from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow import keras
from werkzeug.utils import secure_filename
import numpy as np
from keras.utils import load_img, img_to_array
from keras.models import load_model
import os

app = Flask(__name__)

# Load the saved model
model = load_model('ball_classifier.h5')
train_dir = 'data/train'
class_labels = sorted(os.listdir(train_dir))

def classify_image(file_path):
    # Make a prediction on the uploaded file
    img = load_img(file_path, target_size=(224, 224))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)[0]
    predicted_class_index = np.argmax(prediction)
    predicted_class_label = class_labels[predicted_class_index]
    return predicted_class_label

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_file():
    # Get the uploaded file
    file = request.files['file']

    # Save the file to disk
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # Classify the uploaded file
    predicted_class_label = classify_image(file_path)

    # Delete the uploaded file from disk
    os.remove(file_path)

    # Render the index template with the predicted class label
    return render_template('index.html', predicted_class_label=predicted_class_label)

if __name__ == '__main__':
    app.run()
