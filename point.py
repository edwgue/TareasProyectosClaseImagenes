import cv2
import json
import numpy as np
import sys
import os
from matplotlib import pyplot as plt



data = 30 + 5*np.random.rand(1, 1000)
data = data.flatten()

print (data)
N = 10
min=np.amin(data)
print (min)
max=np.amax(data)
print (max)

step = (max-min)/N

celda_ini=min
celda_fin= min*step


print (sum(data < celda_fin))

steps = np.arange(min,max, step)
print (steps)


