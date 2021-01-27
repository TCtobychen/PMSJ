import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
import os
from functools import wraps
import time
import sys

INPUT_DIM = 131
OUTPUT_DIM = 5
TRIAL_CNT = 0
OUTPUT_LOSS_INTERVAL = 5000

np.random.seed(2333333)

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

def define_model():
    input_dim = INPUT_DIM
    '''
    hidden_dim1 = trail.suggest_int("hidden_dim1", 100, 200)
    hidden_dim2 = trail.suggest_int("hidden_dim2", 100, 200)
    hidden_dim3 = trail.suggest_int("hidden_dim3", 100, 200)
    non_linear_activation = trail.suggest_categorical("non_linear_activation", ["ReLU", "Sigmoid"])
    '''
    hidden_dim1 = 200
    hidden_dim2 = 200
    non_linear_activation = "ReLU"
    output_dim = OUTPUT_DIM
    Model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim1),
        getattr(nn, non_linear_activation)(),
        nn.Linear(hidden_dim1, hidden_dim2),
        getattr(nn, non_linear_activation)(),
        nn.Linear(hidden_dim2, output_dim),
    )
    #print(TRIAL_CNT, hidden_dim1, hidden_dim2, hidden_dim3, non_linear_activation)
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
    
class Args:
    def __init__(self):
        self.batch_size = 64
        self.fold_num = 3
        
args = Args()
device = 'cpu'

def loadData(args):
    data = pd.read_csv("../input/jane-street-market-prediction/train.csv")#, skiprows = lambda x: x>0 and np.random.rand() > 0.01)
    data = data.fillna(0.0)
    dates = np.array(data.date.value_counts().index)
    np.random.shuffle(dates)
    fold_len = int(len(dates)/4)
    test_dates = [dates[i] for i in range(args.fold_num * fold_len, (args.fold_num+1) * fold_len)]
    train_dates = [date for date in dates if date not in test_dates]
    feature_cols = [col for col in data.columns if 'feature' in col]
    feature_cols.append('weight')
    target_cols = [col for col in data.columns if 'resp' in col]
    x_train, y_train, x_test, y_test = {}, {}, {}, {}
    x_train_tensor, y_train_tensor = [], []
    print('num of train dates: ', len(train_dates))
    print('num of test dates: ', len(test_dates))
    for date in train_dates:
        x_train[date] = np.array(data.query(f'date=={date}').loc[:,feature_cols])
        y_train[date] = np.array(data.query(f'date=={date}').loc[:,target_cols])
        x_train_tensor.append(x_train[date])
        y_train_tensor.append(y_train[date])
    for date in test_dates:
        x_test[date] = np.array(data.query(f'date=={date}').loc[:,feature_cols])
        y_test[date] = np.array(data.query(f'date=={date}').loc[:,target_cols])
    x_train_tensor = np.concatenate(x_train_tensor, axis=0)
    y_train_tensor = np.concatenate(y_train_tensor, axis=0)
    np.random.shuffle(x_train_tensor)
    np.random.shuffle(y_train_tensor)
    x_train_tensor = torch.from_numpy(x_train_tensor).float().to(device)
    y_train_tensor = torch.from_numpy(y_train_tensor).float().to(device)
    return x_train, y_train, x_test, y_test, x_train_tensor, y_train_tensor

def get_statistics(model, thres=0.0005):
    y_pred = predict(model, x_test)
    action_test = y_test[:,-1] > thres
    action_pred = y_pred[:,-1] > thres
    return np.sum((action_test-action_pred)==0)/len(action_test)

def predict(model, x):
    x = torch.from_numpy(x).float().to(device)
    return model(x).cpu().detach().numpy()

def measure(model, thres=0.0005):
    pnls = []
    for date, x in x_test.items():
        y_pred = predict(model, x)
        action_pred = y_pred[:,-1] > thres 
        day_pnl = action_pred.reshape(-1) * x[:,-1].reshape(-1) * y_test[date][:,-1].reshape(-1)
        pnls.append(np.sum(day_pnl))
    sharpe = np.sum(pnls)/np.sum(np.square(pnls)) * np.sqrt(250/len(pnls))
    print(sharpe, np.sum(pnls))
    print(pnls)
    return min(max(sharpe, 0), 6) * np.sum(pnls)
    
@timethis
def objective():
    Model = define_model().to(device)
    lr = 1e-4
    #lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    optimizer = optim.Adam(Model.parameters(), lr=lr)
    #n_epochs = trial.suggest_int("epochs", 10, 30)
    n_epochs = 50
    loss_fn = nn.MSELoss(reduction = "mean")
    train_step = make_train_step(Model, loss_fn, optimizer, args)
    #n_epochs = 1
    for epoch in range(n_epochs):
        loss = train_step(x_train_tensor, y_train_tensor, epoch)
    #statistics = get_statistics(Model)
    global TRIAL_CNT
    TRIAL_CNT += 1
    #print(TRIAL_CNT, statistics)
    metrics = measure(Model)
    
    return Model

x_train, y_train, x_test, y_test, x_train_tensor, y_train_tensor = loadData(args)
model = objective()

import janestreet
from tqdm import tqdm
env = janestreet.make_env()
for (test_df, pred_df) in tqdm(env.iter_test()):
    if test_df['weight'].item() > 0:
        test_df = test_df.fillna(0.0)
        feature_cols = [col for col in test_df.columns if 'feature' in col]
        feature_cols.append('weight')
        target_cols = [col for col in test_df.columns if 'resp' in col]
        x_tt = np.array(test_df.loc[:, feature_cols]).reshape(-1, INPUT_DIM)
        x_tt_tensor = torch.from_numpy(x_tt).float().to(device)
        y_pred = model(x_tt_tensor).cpu().detach().numpy().reshape(-1)
        pred_df.action = 1 if y_pred[-1] > 0.0005 else 0
    else:
        pred_df.action = 0
    env.predict(pred_df)