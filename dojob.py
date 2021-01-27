import numpy as np 
import pandas as pd
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import argparse
import os
from jump_tools import *

parser = argparse.ArgumentParser(description='JS Market Prediction')
parser.add_argument('--fold-num', type=int, default=0,
                    help='which fold to test, choose from 0 to 3')
parser.add_argument('--batch-size', type=int, default=64,
                    help='batch size')
args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

def loadData(args):
    data = pd.read_csv("./train.csv")#, skiprows = lambda x: x>0 and np.random.rand() > 0.01)
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
def objective(trial):
    Model = define_model(trial).to(device)
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    optimizer = optim.Adam(Model.parameters(), lr=lr)
    n_epochs = trial.suggest_int("epochs", 10, 30)
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
    
    return metrics

# x, y as numpy arrays
x_train, y_train, x_test, y_test, x_train_tensor, y_train_tensor = loadData(args)

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    
    print("Results:", study.best_params)
    print(study.best_value)
    print(study.best_trial)