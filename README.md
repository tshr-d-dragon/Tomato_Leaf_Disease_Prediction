# Tomato_Leaf_Disease_Prediction
 
### A DL project with deployment to predict tomato leaf disease using transfer learning techniques

This projects helps predicting sign language gestures. 
I tried various transfer learning models for training and chose Densenet121 because of comparatively better performance and smaller size benefitial for smoother deployment.

## Project Structure
1. Tomato_transfer_learning_densenet121.ipynb file gives the walkthrough over the complete project. Weights for all models trained are stored in the models/model_weights folder. The models folder also contain all the ipynb files giving the walkthrough over training of all models at models/model_training location.
2. label.txt file contains all the 10 classes to be predicted with model.
3. Predicted_Images folder contains all the predicted images and label_save.txt stores its prediction value along with probability of prediction.
4. app.py file gives the walkthrough over the deployment of project in flask. All the required templates are stored in templates folder. The info.ini file contains information shown in after prediction.
5. test_images folder contains images that can be used for training.

## To run the prject, follow below steps
1. Ensure that you are in the project home directory
2. Create anaconda environment
3. Activate environment
4. >pip install -r requirement.txt
5. >python app.py
6. Navigate to URL http://localhost:5000

## Please feel free to connect for any suggestions or doubts!!!

## Credits
1. The credits for dataset used for training goes to https://www.kaggle.com/noulam/tomato
2. I have modified https://github.com/Pawandeep-prog/resnet-flask-webapp/tree/main/templates html templates for flask
3. The credit for image used in html file for background goes to: 
  
  a. https://gray-kfyr-prod.cdn.arcpublishing.com/resizer/FQLjpb-2QAmZen2144Gpj79B7tI=/1200x675/smart/cloudfront-us-east-1.images.arcpublishing.com/gray/FVYN3SW6F5FBHOQNSAPYCK2CPI.jpeg
  
  b. https://www.thespruce.com/thmb/47xukLrGeP6r8jbmyeFFujXn4ug=/960x0/filters:no_upscale():max_bytes(150000):strip_icc():format(webp)/top-tomato-growing-tips-1402587-11-c6d6161716fd448fbca41715bbffb1d9.jpg
  
##### For better prediction, we need better image quality dataset for training.
