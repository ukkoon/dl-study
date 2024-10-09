import numpy as np
import pandas as pd

def cos_sim1 (x,y, eps=1e-8):
    nx = x/np.sqrt(np.sum(x**2)+eps)    
    ny = y/np.sqrt(np.sum(y**2)+eps)
    # print(f"nx:{nx}")
    # print(f"ny:{ny}")
    xy_cos_sim = np.dot(nx,ny)
    return xy_cos_sim    

# definition-style
def cos_sim2(x,y, eps=1e-8):    
    xnorm = np.linalg.norm(x, 2) +eps
    ynorm = np.linalg.norm(y, 2) +eps
    # print(f'xnorm:{xnorm}')
    # print(f'ynorm:{ynorm}')
    xy_cos_sim = np.dot(x,y)/(xnorm*ynorm)
    return xy_cos_sim

