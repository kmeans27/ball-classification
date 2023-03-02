import os
import csv
from datetime import datetime
from tensorflow import keras
import numpy as np
from github import Github
from github import InputFileContent

# GitHub repository information
owner = 'kmeans27'
repo = 'ball-classification'
branch = 'main'
path = 'output'

# GitHub API token
token = os.environ.get("PAT")

# Set up paths and directories
image_dir = 'new_images'
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
csv_path = os.path.join(output_dir, 'predictions.csv')

# Set environment variable to use CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Load the saved model
model = keras.models.load_model('ball_classifier.h5')
train_dir = 'data/train'
class_labels = sorted(os.listdir(train_dir))

def classify_image(file_path):
    # Make a prediction on the uploaded file
    img = keras.utils.load_img(file_path, target_size=(224, 224))
    img = keras.utils.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)[0]
    predicted_class_index = np.argmax(prediction)
    predicted_class_label = class_labels[predicted_class_index]
    return predicted_class_label


# Get the list of image files
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

# Make predictions for each image and save to CSV
with open(csv_path, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['filename', 'predicted_class_label', 'timestamp'])
    for file in image_files:
        file_path = os.path.join(image_dir, file)
        predicted_class_label = classify_image(file_path)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([file, predicted_class_label, timestamp])

# Push changes to GitHub
g = Github(token)
repo = g.get_repo(f"{owner}/{repo}")
branch = repo.get_branch(branch)
file_name = "predictions.csv"
file_path = f"{path}/{file_name}"
contents = repo.get_contents(file_path, ref=branch.name)
with open(csv_path, "r") as file:
    content = file.read()
if contents:
    repo.update_file(contents.path, f'Update {file_name}', content, contents.sha, branch=branch.name)
else:
    repo.create_file(file_path, f'Add {file_name}', content, branch=branch.name)
