from numba import complex128, float64, uint16
from cmath import exp
from numba import cuda
from math import log as real_log


@cuda.jit('void(uint16[:,:], complex128, float64, float64, float64, complex128, float64)')
def rational2_2(image_array, topleft, xstride, ystride, lambd, julia, rot):
    y, x = cuda.grid(2)

    if x < image_array.shape[1] and y < image_array.shape[0]:
        z = complex128(topleft + x * xstride - 1j * y * ystride) * exp(1j*rot)

        i = 0
        while i < 1000 and z.real * z.real + z.imag * z.imag < 4:
            z = z * z - lambd / (z*z) + julia
            i += 1
        k = real_log(float64(i)) / real_log(999.0)
        norm = z.real * z.real + z.imag * z.imag
        if norm>4:
            l = uint16(real_log(z.real * z.real + z.imag * z.imag - 4)*200)
#            if l>255: l = 255
        image_array[y, x] = uint16(255 * k)*256 - l

        

    