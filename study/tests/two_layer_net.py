import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정

from matplotlib import pyplot as plt
import numpy as np
from dataset.mnist import load_mnist
from functions.numerical_gradient import numerical_gradient1
from functions.gradient_descent import gradient_descent
from functions.utils import img_show

from networks.two_layer_net import TwoLayerNet
from PIL import Image

# (x_train, t_train), (x_test, t_test) = load_mnist(normalize=False,one_hot_label=True)
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True,one_hot_label=True)

print(f'x_train.shape:{x_train.shape}')
print(f't_train.shape:{t_train.shape}')
print(f'x_test.shape:{x_test.shape}')
print(f't_test.shape:{t_test.shape}')

# before show image, check whether the data has been normalized or not.
# x_train0_img = x_train[0].reshape(28,28)
# x_train0_label = np.argmax(t_train[0])
# img_show(x_train0_img)

data_count = 100
input_size = 784
hidden_size = 50
label_class_count = 10 # 0~9
train_size = x_train.shape[0]
batch_size = 100
iters_num = 10000
lr = 0.1

net = TwoLayerNet(input_size,hidden_size,output_size=label_class_count)

count = 0

train_acc_list = []
test_acc_list = []


iter_per_epoch = max(train_size/batch_size,1)

# predict_y = net.predict(x_train)
# print(predict_y.shape)
# print(t_train.shape)

for i in range(iters_num):            
    batch_mask = np.random.choice(train_size,batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grads = net.gradient_descent2(x_train[batch_mask],t_train[batch_mask])

    for key in ('W1','b1','W2','b2'):
        net.params[key] -= lr * grads[key]

    if i % iter_per_epoch==0:
        train_acc = net.accuracy(x_train,t_train)
        train_acc_list.append(train_acc)

        test_acc = net.accuracy(x_test,t_test)
        test_acc_list.append(test_acc)
        print(f"train acc:${train_acc} | test acc:${test_acc}")

    count+=1
    
    


x = np.arange(0,len(train_acc_list))

plt.plot(x,train_acc_list,label="train_acc")
plt.plot(x,test_acc_list,'cs--',label="test_acc")

plt.show()