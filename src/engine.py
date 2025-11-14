# src/engine.py
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from .models import MLP, GNN, FinalGNN, T_Learner_GNN
import numpy as np 

# --- Training Utilities (Unchanged) ---
def train_model(model, X, y, train_idx, is_binary=False, **training_kwargs):
    lr = training_kwargs.get('lr', 1e-3); 
    epochs = training_kwargs.get('nuisance_epochs', 150)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss() if is_binary else nn.MSELoss()
    for _ in range(epochs): 
        model.train(); 
        opt.zero_grad(); 
        pred = model(X).squeeze(); 
        loss = loss_fn(pred[train_idx], y[train_idx]); 
        loss.backward(); 
        opt.step()
    return model

def train_gnn_model(model, X, y, edge_index, train_idx, is_binary=False, T=None, **training_kwargs):
    lr = training_kwargs.get('lr', 1e-3); epochs = training_kwargs.get('nuisance_epochs', 150)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss() if is_binary else nn.MSELoss()
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        if T is not None: pred = model(X, T, edge_index).squeeze()
        else: pred = model(X, edge_index).squeeze()
        loss = loss_fn(pred[train_idx], y[train_idx]); loss.backward(); opt.step()
    return model

# --- Nuisance & CATE Estimators (Now fully flexible) ---

def get_nuisance_predictions(X, T, Y, edge_index=None, use_gnn=False, folds=2, 
                             model_kwargs={}, training_kwargs={}):
    n, kf = len(X), KFold(n_splits=folds, shuffle=True, random_state=42)
    Y_hat, T_hat = torch.zeros(n), torch.zeros(n)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(range(n))):
        train_idx = torch.tensor(train_idx, dtype=torch.long)
        val_idx = torch.tensor(val_idx, dtype=torch.long)
        if use_gnn:
            outcome_model = train_gnn_model(GNN(X.shape[1], **model_kwargs), X, Y.squeeze(), edge_index, train_idx, False, **training_kwargs)
            treat_model = train_gnn_model(GNN(X.shape[1], **model_kwargs), X, T.squeeze(), edge_index, train_idx, True, **training_kwargs)
            with torch.no_grad():
                outcome_model.eval(); treat_model.eval()
                Y_hat[val_idx] = outcome_model(X, edge_index).squeeze()[val_idx]
                T_hat[val_idx] = torch.sigmoid(treat_model(X, edge_index).squeeze())[val_idx]
        else: 
            outcome_model = train_model(MLP(X.shape[1], **model_kwargs), X, Y.squeeze(), train_idx, False, **training_kwargs)
            treat_model = train_model(MLP(X.shape[1], **model_kwargs), X, T.squeeze(), train_idx, True, **training_kwargs)
            with torch.no_grad():
                outcome_model.eval(); treat_model.eval()
                Y_hat[val_idx] = outcome_model(X[val_idx]).squeeze()
                T_hat[val_idx] = torch.sigmoid(treat_model(X[val_idx]).squeeze())
    return Y_hat, T_hat

def estimate_cate_linear(Y_res, T_res, X):
    X_np, T_res_np, Y_res_np = [t.detach().numpy() for t in [X, T_res, Y_res]]
    reg = Ridge(alpha=0.1); reg.fit(X_np * T_res_np[:, np.newaxis], Y_res_np)
    return torch.from_numpy(reg.predict(X_np))

def estimate_cate_gnn(Y_res, T_res, X, edge_index, model_obj=None, model_kwargs={}, training_kwargs={}):
    lr = training_kwargs.get('lr', 1e-3); epochs = training_kwargs.get('cate_epochs', 200)
    model = model_obj if model_obj is not None else FinalGNN(X.shape[1], **model_kwargs)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train(); opt.zero_grad()
        tau_hat_gnn = model(X, edge_index).squeeze()
        predicted_y_res = T_res.squeeze() * tau_hat_gnn
        loss = F.mse_loss(predicted_y_res, Y_res.squeeze())
        loss.backward(); opt.step()
    with torch.no_grad(): return model(X, edge_index).squeeze()

def estimate_cate_tlearner_gnn(X, T, Y, edge_index, model_kwargs={}, training_kwargs={}):
    epochs = training_kwargs.get('nuisance_epochs', 150) 
    model = T_Learner_GNN(X.shape[1], **model_kwargs)
    train_idx = torch.arange(len(X))
    model = train_gnn_model(model, X, Y.squeeze(), edge_index, train_idx, False, T=T, epochs=epochs, **training_kwargs)
    with torch.no_grad():
        y1_hat = model(X, torch.ones_like(T), edge_index).squeeze()
        y0_hat = model(X, torch.zeros_like(T), edge_index).squeeze()
    return y1_hat - y0_hat
