import numpy as np
import numpy.linalg as linalg

x1 = [1, 2, 3]

x2 = np.multiply(2, x1)
x3 = 3.5 * x2
x4 = x3 + x2
print("x1 = ", x1)
print("x2 = ", x2)
print("x2[1] = ", x2[1])
print("x3 = ", x3)
print("x4 = ", x4)

x5 = linalg.norm(x1)
print('x5 = ', x5)