import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정

import numpy as np
from functions.numerical_gradient import numerical_gradient1, numerical_gradient2
from functions.gradient_descent import gradient_descent
from networks.simple_net import SimpleNet

net = SimpleNet()

x = np.array ( [[ 0.6, 0.9 ],[1,2],[3,4] ] )
t = np.array ( [[ 0, 1, 0 ],[ 0, 0, 1 ],[ 0, 1, 0 ]])

preds = net.predict(x)
print(net.loss(x,t))
# print(preds)

f = lambda W : net.loss ( x, t )
dw, dw_history = gradient_descent(f,net.W,step_num=10000)

preds = net.predict(x)
print(net.loss(x,t))
# print(preds)

