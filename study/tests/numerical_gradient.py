import os
import sys


current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정

import numpy as np
from functions.numerical_gradient import numerical_gradient1,  numerical_gradient2
from functions.numerical_diff import numerical_diff


def func_1(inputs):
    return np.sum(inputs**2)    

def func_2(a,b):
    return a**2 + b **2

x = np.array([[3.,4.],[5.,6.],[7.,8.]])

print(numerical_diff(func_1,3))
print(numerical_diff(func_1,4))
print(numerical_gradient1(func_1,x))
print(numerical_gradient2(func_2,x))