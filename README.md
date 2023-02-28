# Documentation 

1. Image Classification Model (last_try.ipynb) - trained on 30 different types of balls. Accuracy 72%
2. Save the model as .h5 file (ball_classifier.h5)
3. Write a sample Flask application (flask-app.py), where one can visit a website and upload an (ball) image to get the predicted ball type 
 </br> a) Uses the pretrained .h5 model
 </br> b) Uses the index.html template
 </br> c) Defined a function classify_image() which expects an image file as parameter and returns the predicted ball type
 </br> Missing: option to upload files to the later used: new_images folder to automatically perform the classification in batches
4. Python script which allows to analyze images in batches
 </br> a) uses the classify_image() function
 </br> b) write the predictions to the predictions.csv file
5. Created a actions.yml file to execute the script every night
6. installed git LFS to upload ball-classifier.h5 as the file is larger than 100MB
