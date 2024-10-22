import os
import re
from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


import numpy as np
import matplotlib.pyplot as plt



class SimpleNN(nn.Module):
    def __init__(self, input_dim = 6, h_dims = None):
        super(SimpleNN, self).__init__()
        
        if h_dims == None:
            h_dims = [32, 64, 64, 32]
        
        
        fc_layers = nn.Sequential(
        )
        fc_layers.append(nn.Linear(input_dim, h_dims[0]))
        fc_layers.append(nn.ReLU())
        for i in range(len(h_dims)-1):
            fc_layers.append(nn.Linear(h_dims[i], h_dims[i+1]))
            fc_layers.append(nn.ReLU())
        fc_layers.append(nn.Linear(h_dims[-1], 1))
        
        self.model = nn.Sequential(*fc_layers)
        

    def forward(self, x):
        x = self.model(x)
        return x
    

    def train_model(self, X_train, Y_train, epochs=100, lr=0.001):
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        Y_tensor = torch.tensor(Y_train, dtype=torch.float32).view(-1, 1)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        for epoch in tqdm(range(epochs)):
            self.train()  
            optimizer.zero_grad()  
            outputs = self.forward(X_tensor)  
            loss = criterion(outputs, Y_tensor)  
            loss.backward()  
            optimizer.step()  

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    def evaluate(self, X_test, Y_test):
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_test, dtype=torch.float32)
            Y_tensor = torch.tensor(Y_test, dtype=torch.float32).view(-1, 1)
            outputs = self.forward(X_tensor)
            loss = nn.MSELoss()(outputs, Y_tensor)
            print(f'Test Loss: {loss.item():.4f}')
            return outputs.numpy(), Y_tensor.numpy()


def train_test_split_data(X, Y, test_size=0.3):
    
    num_samples = X.shape[0]
    num_test_samples = int(num_samples * test_size)
    
    
    
    indices = np.random.permutation(num_samples)
    test_indices = indices[:num_test_samples]
    train_indices = indices[num_test_samples:]
    
    
    X_train, X_test = X[train_indices], X[test_indices]
    Y_train, Y_test = Y[train_indices], Y[test_indices]
    print(f"split dataset.  train={num_samples - num_test_samples}, test={num_test_samples}")
    
    return X_train, X_test, Y_train, Y_test


def load_data_from_txt(fpath):
    
    with open(fpath, "r") as f:
        lines = f.readlines()
        
    X_list, Y_list = [], []
    for line in lines:
        numbers = [float(num) for num in re.findall(r"-?\d+\.?\d*", line)]
        X_list.append(numbers[:-1]) 
        Y_list.append(numbers[-1])
    
    
    X = np.array(X_list)  ## (len, 6)
    Y = np.expand_dims(np.array(Y_list), axis = -1) ## (len, 1)
    return X, Y


def main():

    # 데이터
    X, Y = load_data_from_txt(fpath = "./std.txt")
    X_train, X_test, Y_train, Y_test = train_test_split_data(X, Y)
    
    

    # 모델 초기화 및 학습
    model = SimpleNN()
    model.train_model(X_train, Y_train, epochs=100)

    # 모델 평가
    model.evaluate(X_test, Y_test)
    
    
    



if __name__ == "__main__":
    main()