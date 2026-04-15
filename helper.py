
import numpy as np 
import pandas as pd 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


from tqdm import tqdm

# let's visualize a dog maybe 




# let's visualize adoption rates 





# Models

# show pred vs actual confusion matrix for MLP model 
def run_mlp_pipeline(pet_qs): 
    # i did NOT feel like writing out reason 1 through 5 lol 
    questions = ["reason" + str(i) for i in range(1, 6)] + ["replace" + str(i) for i in range(1, 4)]

    for col in questions: 
        pet_qs[col] = pet_qs[col].map({"Yes": 2, "No": 1})

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
        random_state=42, # keep   
    )

    # scale
    scaler = MinMaxScaler()
    # guys i don't remmber if u scale the x train and test sep or tg but we do sep horray 
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    # training model
    eta = 0.01
    q = 64

    d = X_train_scaled.shape[1]

    W1 = np.random.randn(d, q) * 0.5
    b1 = np.random.randn(q, 1) * 0.1
    W2 = np.random.randn(q, 1) * 0.5
    b2 = np.random.randn(1, 1) * 0.1

    def sigmoid(x): 
        return 1 / (1 + np.exp(-x))

    errors = []
    epochs = 10000
    n = X_train_scaled.shape[0]
    d = X_train_scaled.shape[1]

    print(f"BEFORE training - W1 sum: {W1.sum():.4f}")

    for epoch in tqdm(range(epochs)):
        dW2 = 0
        db2 = 0
        for i, j in enumerate(y_train):
            x = np.reshape(X_train_scaled[i], (d,1))
            h = np.maximum(0, W1.T.dot(x) + b1)
            y_hat = sigmoid(W2.T.dot(h) + b2)
            
            error_term = (y_hat - y_train[i])
            dW2 += (1/n) * error_term * h
            db2 += (1/n) * error_term

        W2 = W2 - eta * dW2
        b2 = b2 - eta * db2
        
        dW1 = 0
        db1 = 0
        for i, j in enumerate(y_train):
            x = np.reshape(X_train_scaled[i], (d,1))
            h = np.maximum(0, W1.T.dot(x) + b1)
            y_hat = sigmoid(W2.T.dot(h) + b2)
            mat1 = np.heaviside(W1.T.dot(x) + b1, 0)

            dW1 += (1/n) * (y_hat - y_train[i]) * np.kron(x, (W2 * mat1).T)
            db1 += (1/n) * (y_hat - y_train[i]) * (W2 * mat1)

        W1 = W1 - eta * dW1
        b1 = b1 - eta * db1
        
        preds_train = []
        for i in range(n):
            x = X_train_scaled[i].reshape(d,1)
            h = np.maximum(0, W1.T.dot(x) + b1)
            preds_train.append(sigmoid(W2.T.dot(h) + b2)[0,0])
        
        e = np.mean((np.array(preds_train) - y_train)**2)
        errors.append(e)

    print(f"AFTER training - W1 sum: {W1.sum():.4f}")
    print(f"First error: {errors[0]:.4f}")
    print(f"Last error: {errors[-1]:.4f}")

    y_test_pred_prob = []
    y_test_pred = []

    for i in range(X_test_scaled.shape[0]):
        x = X_test_scaled[i].reshape(d,1)          
        h = np.maximum(0, W1.T.dot(x) + b1)
        y_hat = sigmoid(W2.T.dot(h) + b2)
        y_test_pred_prob.append(y_hat[0,0]) 
        y_test_pred.append(int(y_hat[0][0] >= 0.5))

    y_test_pred_prob = np.array(y_test_pred_prob)
    y_test_pred = np.array(y_test_pred)


    return y_test_pred, y_test
    