import numpy as np 
import pandas as pd
from functools import wraps
import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import sys

INPUT_DIM = 131
OUTPUT_DIM = 5
TRIAL_CNT = 0
OUTPUT_LOSS_INTERVAL = 100

np.random.seed(2333333)

def preprocessing(train):
    X = train.loc[:, train.columns.str.contains('feature')]
    # y_train = train.loc[:, 'resp']
    Y = train.loc[:, 'action']
    
    x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=666, test_size=0.2)
    
    return x_train, x_test, y_train, y_test 

def score(model, data):
    pass

def choose_features(df):
    pass


def predict_js(model):
    import janestreet
    env = janestreet.make_env()
    iter_test = env.iter_test()
    for (test_df, sample_prediction_df) in iter_test:
        if test_df['weight'].item() > 0:
            test_df = featureEngineer(test_df)
            X_test = choose_features(test_df)
            #X_test = test_df.loc[:, test_df.columns.str.contains('feature')]
            #X_test = X_test.fillna(0.0)
            y_pred = predict(X_test, model)
        else:
            y_pred = 0
        sample_prediction_df.action = y_pred
        env.predict(sample_prediction_df)

def define_model(trail):
    input_dim = INPUT_DIM
    hidden_dim1 = trail.suggest_int("hidden_dim1", 100, 200)
    hidden_dim2 = trail.suggest_int("hidden_dim2", 100, 200)
    hidden_dim3 = trail.suggest_int("hidden_dim3", 100, 200)
    non_linear_activation = trail.suggest_categorical("non_linear_activation", ["ReLU", "Sigmoid"])
    output_dim = OUTPUT_DIM
    Model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim1),
        getattr(nn, non_linear_activation)(),
        nn.Linear(hidden_dim1, hidden_dim2),
        getattr(nn, non_linear_activation)(),
        nn.Linear(hidden_dim2, hidden_dim3),
        getattr(nn, non_linear_activation)(),
        nn.Linear(hidden_dim3, output_dim),
    )
    print(TRIAL_CNT, hidden_dim1, hidden_dim2, hidden_dim3, non_linear_activation)
    return Model

def make_train_step(model, loss_fn, optimizer, args):
    def train_step(x, y, epoch):
        model.train()
        running_loss = []
        for itr in range(len(x)//args.batch_size):
            optimizer.zero_grad()
            yhat = model(x[itr*args.batch_size:(itr+1)*args.batch_size])
            #yhat = yhat.squeeze(-1)
            #print(yhat.shape, x.shape, y[itr*args.batch_size:(itr+1)*args.batch_size, :].shape)
            if itr == -1:
                temp_yhat = yhat.cpu().detach().numpy()
                temp_x = x[itr*args.batch_size:(itr+1)*args.batch_size].cpu().detach().numpy()
                for j in range(len(temp_yhat)):
                    if np.sum(np.isnan(temp_yhat[j])) > 0:
                        print(temp_x[j])
            loss = loss_fn(yhat, y[itr*args.batch_size:(itr+1)*args.batch_size, :])
            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())
            if (itr+1)%OUTPUT_LOSS_INTERVAL == 0:
                rl = np.mean(running_loss)
                print(f'Epoch {epoch}, Iteration {itr}, running_loss: {rl}')
                running_loss = []
    return train_step

def timethis(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        r = func(*args, **kwargs)
        end = time.perf_counter()
        print('{}.{}的运行时间为 : {}秒'.format(func.__module__, func.__name__, end - start))
        return r
    return wrapper
    
