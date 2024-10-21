import os
from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


import numpy as np
import matplotlib.pyplot as plt



class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(6, 32)  # 첫 번째 레이어
        self.fc2 = nn.Linear(32, 32)  # 두 번째 레이어
        self.fc3 = nn.Linear(32, 32)  # 세 번째 레이어
        self.fc4 = nn.Linear(32, 32)  # 네 번째 레이어
        self.fc5 = nn.Linear(32, 1)    # 출력 레이어

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

    def train_model(self, X_train, Y_train, epochs=100, lr=0.001):
        # 데이터 텐서로 변환
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        Y_tensor = torch.tensor(Y_train, dtype=torch.float32).view(-1, 1)

        # 손실 함수 및 옵티마이저 정의
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        for epoch in tqdm(range(epochs)):
            self.train()  # 모델을 학습 모드로 설정
            optimizer.zero_grad()  # 기울기 초기화
            outputs = self(X_tensor)  # 예측
            loss = criterion(outputs, Y_tensor)  # 손실 계산
            loss.backward()  # 역전파
            optimizer.step()  # 가중치 업데이트

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    def evaluate(self, X_test, Y_test):
        # 평가 모드로 설정
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_test, dtype=torch.float32)
            Y_tensor = torch.tensor(Y_test, dtype=torch.float32).view(-1, 1)
            outputs = self(X_tensor)
            loss = nn.MSELoss()(outputs, Y_tensor)
            print(f'Test Loss: {loss.item():.4f}')
            return outputs.numpy(), Y_tensor.numpy()

# def train_test_split_data(X, Y, test_size=0.3):
#     return train_test_split(X, Y, test_size=test_size, random_state=42)

def main():
    X = np.array([[1, 2, 3, 4, 5, 6],
              [2, 3, 4, 5, 6, 7],
              [3, 4, 5, 6, 7, 8]])
    Y = np.array([10, 20, 30])

    # 데이터 분할
    # X_train, X_test, Y_train, Y_test = train_test_split_data(X, Y)
    X_train, X_test, Y_train, Y_test = X, X, Y, Y

    # 모델 초기화 및 학습
    model = SimpleNN()
    model.train_model(X_train, Y_train, epochs=100)

    # 모델 평가
    model.evaluate(X_test, Y_test)
    
    
    



if __name__ == "__main__":
    main()