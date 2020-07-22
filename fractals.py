from base import create_image_array, create_image, random_image
from kernels import rational2_2
from math import log10
from matplotlib.pyplot import imshow
import numpy as np

"""
jscal = 100
lambdo = -3
lambdr = 3
juliao = 10**lambdo * jscal / 100
juliar = 1
rot = 12/180*pi

for i in range(0,10,10):
    lambd = 10**(lambdo - lambdr*i/100)
    juli = rect(juliao, juliar*i/100*pi)
    create_image(rational2_2, -0.7, 0.7, -0.7, 0.7, 1000, 1000, lambd, juli, rot)
"""
while (True):
    ima = random_image(rational2_2, 
                       -0.7, 0.7, -0.7, 0.7,
                       log10(0.01), log10(1), 
                       log10(0.003), log10(0.03), 
                       log10(1), log10(10))
    if np.sum(ima>65000)<1000: break
ima = np.random.normal(ima,300)
imshow(ima, cmap='gray')