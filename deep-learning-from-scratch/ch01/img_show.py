# coding: utf-8
import matplotlib.pyplot as plt
from matplotlib.image import imread
import os

img = imread(f'{os.getcwd()}/dataset/cactus.png') # 이미지 읽어오기
plt.imshow(img)

plt.show()
