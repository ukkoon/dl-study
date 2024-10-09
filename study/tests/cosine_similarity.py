import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정

import numpy as np
# import pandas as pd
from functions.cosine_similarity import cos_sim1, cos_sim2 

x = np.array([[1,2,4], [-1,-2,-4], [0,0,0]])
y = np.array([1,2,4])

# file_name = "the-effect-of-calcium-on-iron-absorption"
# datafile_path = f"{file_name}.csv"

# df = pd.read_csv(datafile_path)
# df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)

# contexts = np.array(df)
# question_emedding = np.array([])

target = -1
highst = float("-inf")

for i in range(len(x)):    
    xy_cos_sim = cos_sim1(x[i],y)
    print(f'cos_sim with({x[i]},{y}):{xy_cos_sim}')
    if xy_cos_sim>highst:
        highst = xy_cos_sim
        target = i
