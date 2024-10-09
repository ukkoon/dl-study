import numpy as np

# 모든 변수의 편미분을 벡터로 정리한 것을 기울기(gradient)라고 한다.
# 기울기 구하기 함수
def numerical_gradient1(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 값 복원
        it.iternext()   
        
    return grad

def numerical_gradient2(f, x):
    h = 1e-4 # 0.0001
    grads = np.zeros_like(x)

    # 간단히 x의 인자를 x[idx]라고 할 때, f(x[idx])의 미분을 구하는 간단한 식이라고 이해

    for i in range(len(x)):
        grad = []
        tmp_val = x[i][0]
        x[i][0] = float(tmp_val) + h
        fxh1 = f(x[i][0],x[i][1])

        x[i][0] = tmp_val - h
        fxh2 = f(x[i][0],x[i][1])
        grad.append((fxh1-fxh2)/(2*h))
        
        tmp_val = x[i][1]
        x[i][1] = float(tmp_val) + h
        fxh1 = f(x[i][0],x[i][1])

        x[i][1] = tmp_val - h
        fxh2 = f(x[i][0],x[i][1])
        grad.append((fxh1-fxh2)/(2*h))
        grads[i]=grad        
        
    return grads