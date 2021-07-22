
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os, ssl, time
from flask import Flask, json, jsonify, request

#Fetching the data
X, y = fetch_openml('mnist_784', version=1, return_X_y=True)

#Splitting the data and scaling it
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=9, train_size=7500, test_size=2500)
#scaling the features
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

#Fitting the training data into the model
clf = LogisticRegression(solver='saga', multi_class='multinomial').fit(X_train_scaled, y_train)

def getPrediction(image):
  im_pil = Image.open(image)
  image_bw = im_pil.convert('L')
  image_bw_resized_inverted = image_bw.resize((28,28), Image.ANTIALIAS)
  pixel_filter = 20
  min_pixel = np.percentile(image_bw_resized_inverted, pixel_filter)
  image_bw_resized_inverted_scaled = np.clip(image_bw_resized_inverted-min_pixel, 0, 255)
  max_pixel = np.max(image_bw_resized_inverted)
  image_bw_resized_inverted_scaled = np.asarray(image_bw_resized_inverted_scaled)/max_pixel
  test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1,784)
  test_pred = clf.predict(test_sample)
  return test_pred[0]

app =Flask(__name__)

@app.route('/predict_num', methods = ['POST'])

def predictNum():
  image = request.files.get('digit')
  prediction = getPrediction(image)
  return jsonify({'prediction' : prediction})
  
if(__name__ == '__main__'):
    app.run()