import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle
import os
import cv2
from sklearn.preprocessing import LabelEncoder

INPUT_FILE = "processed_data.csv"
MODEL_FILE = "svm_model.pkl"

def load_data(input_file):
    df = pd.read_csv(input_file, header=0)
    features = df.to_numpy()[:, :-1]

    target_column = df.iloc[:, -1].values
    label_encoder = LabelEncoder()
    target = label_encoder.fit_transform(target_column)

    return train_test_split(features, target, test_size=0.2, random_state=0), label_encoder

def save_model(model, model_file):
    with open(model_file, "wb") as f:
        pickle.dump(model, f)

def load_model(model_file):
    with open(model_file, "rb") as f:
        return pickle.load(f)

def train_or_load_model(x_train, y_train, model_file):
    if os.path.exists(model_file):
        print(f"Loading model from {model_file}")
        return load_model(model_file)
    else:
        print(f"Training model and saving to {model_file}")
        clf = SVC(probability=True)
        clf.fit(x_train, y_train)
        save_model(clf, model_file)
        return clf

def evaluate_model(model, x_test, y_test):
    score = model.score(x_test, y_test)
    print(f"Model accuracy: {score}")

def svm(image):
    (x_train, x_test, y_train, y_test), label_encoder = load_data(INPUT_FILE)
    model = train_or_load_model(x_train, y_train, MODEL_FILE)

    resized_image = cv2.resize(image, (128,128))

    edges = cv2.Canny(resized_image, 100, 200)
    
    edge_vector = edges.flatten()
    predict_res= model.predict(edge_vector.reshape(1, -1))
    predict_proba = model.predict_proba(edge_vector.reshape(1, -1))
    print(str(predict_res)+" "+str(predict_proba))
    res = label_encoder.inverse_transform(predict_res)
    print(res)
    return res[0]+" "+"{:.2f}".format(max(predict_proba[0]))

if __name__ == "__main__":
    (x_train, x_test, y_train, y_test), label_encoder = load_data(INPUT_FILE)
    model = train_or_load_model(x_train, y_train, MODEL_FILE)

    image=cv2.imread("image.png")
    resized_image = cv2.resize(image, (128,128))

    edges = cv2.Canny(resized_image, 100, 200)
    
    edge_vector = edges.flatten()
    # print(label_encoder.inverse_transform(model.predict(edge_vector.reshape(1, -1))))
    print(model.predict_proba(edge_vector.reshape(1, -1)))
