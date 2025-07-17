import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from scipy import optimize
import numpy as np




def parse_csv():
    file = open('fake_lorenz_data.csv')
    
    data = []
    time = []
    
    for line in file:
        # print(line)
        if line[0] != '#':
            x,y,z,t = line.split(',')
            v = [float(x), float(y), float(z)]
            # print('vector = ', v)
            data.append(v)
            time.append(float(t))
    return (time, data)
def approx_dy(y, time):
    
    data = []
    
    for i in range(1, len(y)):
        v1 = y[i-1]
        v2 = y[i]
        
        dt = time[i] - time[i-1]
        
        x1 = v1[0]
        x2 = v2[0]
        dx = (x1 + x2)/dt
        
        y1 = v1[1]
        y2 = v2[1]
        dy = (y1 + y2)/dt
        
        z1 = v1[2]
        z2 = v2[2]
        dz = (z1 + z2)/dt
        
        v = [dx, dy, dz]
        
        # print('v = ', v)
        data.append(v)
               
    data.append(data[-1])
    return data
def dictionary(v):
    x, y, z = v
    
    a = [
                  1,        x,         y,         z, 
               x**2,    y**2,     z**2,      x*y,    
                x*z,    y*z,   x**3,   y**3, 
             z**3, x**2*y, x**2*z,   x*y*z, 
            y**2*x, y**2*z,  z**2*x,  z**2*y 
        ]
    
    return a
def theta(matrix_data):

    matrix = []
    
    for v in matrix_data:
        column = dictionary(v)
        # print('column = ',column)
        matrix.append(column)
    
    return matrix

def lasso_objective(xi, Theta_X, X_k, lambda_param):
    reconstruction_error = np.linalg.norm(X_k - Theta_X @ xi, 2)
    sparsity_term = lambda_param * np.linalg.norm(xi, 1)
    return reconstruction_error + sparsity_term

# data
data = parse_csv()
time = data[0]
y = data[1]

# approximate derivatives
dy = approx_dy(y, time)


theta_matrix = theta(y)
print('len(theta_matrix) = ', len(theta_matrix))
print('len(theta_matrix[0]) = ', len(theta_matrix[0]))
# print(theta_matrix)



# Solve the optimization problem
'''
result = optimize.minimize(lasso_objective, x0=np.zeros(len(theta_matrix[0])), 
                          args=(np.array(theta_matrix), np.array(dy[0]), 0))
xi_k = result.x
'''