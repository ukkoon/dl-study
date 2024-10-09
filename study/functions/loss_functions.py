import numpy as np

from functions.activation_functions import softmax


def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size    