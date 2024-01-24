# MachineLearningAlgorithums

Overview This Project aims to build application for machine learning algorithms of three models Random forest classifier ,Decision Tree classifier, knn classifier.And slso focusing for predction using random forest model,All the major image classification process,Detail NLP Processing, and customer recommendation process.all the above macjine learning process in one application.

Clone the repository: git clone https://github.com/Sriramgauthamj3004

Install dependencies:

pip install -r requirements.txt

Dependencies Streamlit Pandas NumPy OpenCV Pytesseract PIL (Pillow) Matplotlib Scikit-learn Seaborn Plotly Express NLTK Spacy Wordcloud Surprise

Code Examples: Image Processing:

import streamlit as st import cv2 from PIL import Image import pytesseract

Machine Learning: from sklearn.model_selection import train_test_split from sklearn.ensemble import RandomForestClassifier from surprise import SVD
