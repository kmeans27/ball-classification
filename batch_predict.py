import os
import csv
from datetime import datetime
from flask_app import classify_image

# Set up paths and directories
image_dir = 'new_images'
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
csv_path = os.path.join(output_dir, 'predictions.csv')

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
