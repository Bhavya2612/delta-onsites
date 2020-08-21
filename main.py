''' IMPORTING LIBRARIES '''
import streamlit as st
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.decomposition import PCA
''' TITLE OF THE APP'''
st.title("KNN using Streamlit")

dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris","Breast Cancer","Wine dataset"))
classifier_name = st.sidebar.selectbox("Our classifier is KNN",("KNN","SVM"))
points = st.sidebar.selectbox("Select the number of points you want",("150","100","80","60","40","20"))
def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    X =data.data
    y = data.target
    return X,y

X,y = get_dataset(dataset_name)
st.write("shape of dataset", X.shape)
st.write("number of classes", len(np.unique(y)))

def add_parameter_ui(KNN):
    params = dict()
    if KNN == "KNN":
        K = st.sidebar.slider("K",1,15)
        params["K"] = K
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(KNN,params):
    if KNN == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    return clf

clf = get_classifier(classifier_name, params)
if points == "20":
    X_new = X[0:21,0:4]
    y_new = y[0:21,]
elif points == "40":
    X_new = X[0:41,0:4]
    y_new = y[0:41,]
elif points == "60":
    X_new = X[0:61,0:4]
    y_new = y[0:61,]
elif points == "80":
    X_new = X[0:81,0:4]
    y_new = y[0:81,]
elif points == "100":
    X_new = X[0:101,0:4]
    y_new = y[0:101,]
else:
    X_new = X
    y_new = y

st.write(X_new)

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size = 0.2, random_state = 1234)

clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)

st.write("Classifier is: KNN")
st.write(f"accuracy = {acc}")



#plot
pca  = PCA(2)


X_projected = pca.fit_transform(X_new)


x1 = X_projected[:,0]
x2 = X_projected[:,1]

fig = plt.figure()
plt.scatter(x1,x2,c=y_new, alpha = 0.8,cmap = "viridis" )
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.colorbar()

st.pyplot()
