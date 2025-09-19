import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

st.title("Iris Flower Prediction")

iris = load_iris()
X = iris.data
y = iris.target

clf = RandomForestClassifier()
clf.fit(X, y)

sepal_length = st.slider('Sepal Length', float(X[:, 0].min()), float(X[:, 0].max()))
sepal_width = st.slider('Sepal Width', float(X[:, 1].min()), float(X[:, 1].max()))
petal_length = st.slider('Petal Length', float(X[:, 2].min()), float(X[:, 2].max()))
petal_width = st.slider('Petal Width', float(X[:, 3].min()), float(X[:, 3].max()))

input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = clf.predict(input_data)
st.write(f"Predicted Iris Class: {iris.target_names[prediction][0]}")
