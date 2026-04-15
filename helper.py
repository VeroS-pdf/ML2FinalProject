
from copyreg import pickle

import numpy as np 
import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tqdm import tqdm
import pickle

# let's visualize a dog maybe 






# Models

def run_mlp_pipeline(pet_qs): 
    # i did NOT feel like writing out reason 1 through 5 lol 
    questions = ["reason" + str(i) for i in range(1, 6)] + ["replace" + str(i) for i in range(1, 4)]

    for col in questions: 
        pet_qs[col] = pet_qs[col].map({"Yes": 1, "No": 0})

    # other columns that I care about 
    pet_qs = pd.get_dummies(pet_qs, columns=["sex", "city_YN"], dtype=int)

    features = questions + ["sex_Female", "sex_Male", "trait_pet3", "age", "salary", "city_YN_Rural_area", "city_YN_Urban_area"]
    cleaned = pet_qs[features]
    cleaned = cleaned.fillna(0)

    # now we prep the data for training and stuff 
    y = cleaned["trait_pet3"] # does the person have a pet, and what kind of pet!!! 
    X = cleaned.drop('trait_pet3', axis=1) # DROPPPP It

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2,     
        random_state=24    
    )
    
    def sigmoid(x): 
        return 1 / (1 + np.exp(-np.clip(x, -500, 500))) 

    # scale
    scaler = MinMaxScaler()
    
    # i scale 
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    y_train = y_train.to_numpy()
    y_test = y_test.to_numpy()

    # idk wherre the 2 is coming from stillll 
    y_train = np.where(y_train == 2, 1, 0)
    y_test  = np.where(y_test  == 2, 1, 0)

    # don't train just extract the pickle 
    with open("mlp_model.pkl", "rb") as f:
        model = pickle.load(f)

    d = X_train_scaled.shape[1]

    with open("mlp_model.pkl", "rb") as f:
        model = pickle.load(f)

    W1 = model["W1"]
    b1 = model["b1"]
    W2 = model["W2"]
    b2 = model["b2"]


    y_test_pred_prob = []
    y_test_pred = []

    for i in range(X_test_scaled.shape[0]):
        x = X_test_scaled[i].reshape(d, 1)
        h = np.maximum(0, W1.T.dot(x) + b1)
        y_hat = sigmoid(W2.T.dot(h) + b2)
        y_test_pred_prob.append(y_hat[0, 0])
        y_test_pred.append(int(y_hat[0][0] >= 0.8))

    y_test_pred_prob = np.array(y_test_pred_prob)
    y_test_pred = np.array(y_test_pred)

    return y_test_pred, y_test
    