import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정

import numpy as np
from functions.loss_functions import cross_entropy_error, softmax


class SimpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3) # 정규분포로 초기화
        print("-"*50)
        print("Weigth of SimpleNet")
        print(self.W)
        print("-"*50)   

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss