import numpy as np
import numba as nb
import random as rnd
from math import ceil
from cmath import exp as cexp


from numba import cuda
from PIL import Image



def create_image_array(kernel, xmin, xmax, ymin, ymax, max_iter, base_accuracy, *args):
    if abs(xmax - xmin) > abs(ymax - ymin):
        ny = base_accuracy
        nx = int((base_accuracy * abs(xmax - xmin) / abs(ymax - ymin)))
    else:
        nx = base_accuracy
        ny = int(base_accuracy * abs(ymax - ymin) / abs(xmax - xmin))

    xstride = abs(xmax - xmin) / nx
    ystride = abs(ymax - ymin) / ny
    topleft = nb.complex128(xmin + 1j * ymax)
    image_array = np.zeros((ny, nx), dtype=np.uint8)
    run_kernel(kernel, image_array, topleft, xstride, ystride, max_iter, *args)

    return image_array


def create_image(kernel,
                 xmin, xmax, ymin, ymax,
                 max_iter,
                 base_accuracy,
                 *args,
                 path='fractal.png',
                 show=True):
    image_array = create_image_array(kernel, xmin, xmax, ymin, ymax, max_iter, base_accuracy, *args)

    image = Image.fromarray(image_array)
    if show: 
        image.show()
    else:
        image.save(path, "PNG", quality=95, optimize=True)        
    return image_array


def run_kernel(kernel, image, topleft, xstride, ystride, max_iter, *args):

    dimage = cuda.to_device(image)
    threadsperblock = (32, 16)
    blockspergrid = (ceil(image.shape[0] / threadsperblock[0]), ceil(image.shape[1] / threadsperblock[1]))

    kernel[blockspergrid, threadsperblock](dimage, topleft, xstride, ystride, *args)
    dimage.to_host()


def random_image(kernel, 
                 xmi, xma, ymi, yma,
                 log_scale_min, log_scale_max,
                 log_lambda_min, log_lambda_max,
                 log_julia_min, log_julia_max):
    
    scale = 10**rnd.uniform(log_scale_min, log_scale_max)
    xc = rnd.uniform(xmi, xma)
    yc = rnd.uniform(ymi, yma)
    xmin = xc - scale/2
    ymax = yc + scale/2
    nx = 1000
    ny = 1000
    xstride = scale / nx
    ystride = scale / ny
    topleft = nb.complex128(xmin + 1j * ymax)
    rot = rnd.uniform(0, 360)
    max_iter = 1000
    lambd = 10**rnd.uniform(log_lambda_min, log_lambda_max)
    julia = lambd * 10*rnd.uniform(log_julia_min, log_julia_max) * cexp(1j * rnd.uniform(0,360))
    
    image_array = np.zeros((ny, nx), dtype=np.uint16)
    run_kernel(kernel, image_array, topleft, xstride, ystride, max_iter, lambd, julia, rot)
    
    print(xc, yc, scale, lambd, abs(julia))
    return image_array


