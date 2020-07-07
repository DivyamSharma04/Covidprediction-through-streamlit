import streamlit as st
from PIL import Image
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from keras.models import load_model
import os
import numpy as np
import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True
import joblib

def load_prediction_model(model_files):
	loaded_model = joblib.load(open(os.path.join(model_files),"rb"))
	return loaded_model


st.title("Covid Prediction Through Chest Xray")
st.subheader("This project is only for education purpose, made by Divyam Sharma")
st.markdown("<h1 style='text-align: center; color: ORANGE;'>Please upload yout Chext X Ray</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an image...", type=("jpg","png","jpeg"))
st.text("Please upload chest XRay | only jpg & jpeg & png files")


#showing a sample image
b = st.checkbox("How does a Chest XRay look")
if b:
	Image_1 = Image.open('1.CXRCTThoraximagesofCOVID-19fromSingapore.pdf-000-fig1a.png')
	st.image(Image_1,width=300, caption='Sample Chest XRay')

#showing the uploaded image
a = st.checkbox("Show uploaded image")
if uploaded_file is not None:
		if a:
			Image = Image.open(uploaded_file)
			st.image(Image,width=300, caption='file uploaded by you.')



if st.button("Predict"):

	predictor = load_prediction_model("covid_model.pkl")
	img = image.load_img(uploaded_file, target_size=(224, 224))
	
	# Preprocessing the image
	x = image.img_to_array(img)
	# x = np.true_divide(x, 255)
	x = np.expand_dims(x, axis=0)

	preds = predictor.predict_classes(x)
	preds= np.array_str(preds)
		
	if preds == '0':
		preds = 'Please visit nearby covid centre , it seems you have symptoms'
	else:
		preds='Your X ray seems normal. But in case if you feel any symptoms please visit near by centre'  
    

	final_result = preds
	st.success(final_result)



if __name__ == '__main__':
	main()
