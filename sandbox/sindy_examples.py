import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
'''
t = np.linspace(0, 1, 100)
x = 3 * np.exp(-2 * t)
y = 0.5 * np.exp(t)
X = np.stack((x, y), axis=-1)  # First column is x, second is y\
'''

# fixed for three dimensions
def parse_csv(filename):
    file = open(filename)
    
    data = []
    
    for line in file:
        # print(line)
        if line[0] != '#':
            components = line.split(',')
            v = []
            for c in components:
                v.append(float(c))
            # print('vector = ', v)
            data.append(v)
            # print('t = ', t)
    return data
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
# rhs functions
def rhs_function(vector, t, params, odes):
    new_vector = []
    for i in range(len(vector)):
        # print('ode = ', odes[i])
        new_vector.append(odes[i](vector, t, params))
        # print('new vector = ', new_vector)
    return new_vector 
def rk2_data(y0, n, f, t0, tf, params):
    

    time_data = [t0]
    
    chart_data = []
    
    
    for i in range(len(f)):
        chart_data.append([])
    
    y = np.array(y0)
    t = t0
    dt = (tf-t0)/n
    
    # print('len(f) = ', len(f))
    # print('y = ', y)

    '''
    for i in range(len(f)):
        ode = f[i]
        for j in range(n):
            # print('y[0] = ', y[0])
            y[i] = y[i] + .5 * dt * ode(y, t, param1)
            print('1. y[i] = ', y[i])
            y[i] = y[i] + dt * ode(y, t + .5 * dt, param1)
            print('2. y[i] = ', y[i], '\n')
            chart_data[i].append(y[i])
            # if i == 1:
            #    time_data.append(time_data[len(time_data)-1] + dt) 
    '''
    
    '''
    for j in range(n):
        for i in range(len(f)):
            ode = f[i]
            y[i] = y[i] + .5 * dt * ode(y, t, params)
            # print('1. y[i] = ', y[i])
            y[i] = y[i] + dt * ode(y, t + .5 * dt, params)
            # print('2. y[i] = ', y[i], '\n')
            chart_data[i].append(y[i])
        last_time = time_data[len(time_data) - 1]
        time_data.append(last_time + dt)
    '''
    
    for i in range(n):
        # print("y = ", np.multiply(.5 * dt , collions_rhs(y, t, params, f)))
        # y = np.add(y , np.multiply(.5 * dt , collions_rhs(y, t, params, f)))
        a = np.add(y , np.multiply(.5 * dt , rhs_function(y, t, params, f)))
        y = np.add(y, np.multiply(dt , rhs_function(a, t, params, f)))
        # print("y = ", y)
        last_time = time_data[len(time_data) - 1]
        time_data.append(last_time + dt)
        for j in range(len(chart_data)):
            chart_data[j].append(y[j])
    # print('time data: \n',time_data)
    chart_data.append(time_data[1:])
    return chart_data

def generate_lists(matrix_data):
    
    matrix = []
    for _ in range(len(matrix_data[0])):
        matrix.append([])
    
    for i in range(len(matrix_data)):
        for j in range(len(matrix_data[i])):
            component = matrix_data[i][j]
            matrix[j].append(component)

    return matrix
    
data = parse_csv('fake_lorenz_data.csv')


data = generate_lists(data)
# print(data)

# sigma=10, rho=28, beta=8/3
# x' = 10y -    10x
# y' = 28x -     y - xz
# z' =  xy - 2.667z
x, y, z, time = data

'''
    Lorenz 
'''

'''
X = np.stack((x, y, z), axis=-1)  # First column is x, second is y
model = ps.SINDy(feature_names=["x", "y", "z"])
model.fit(X, t=np.array(time))

print('lorenz: \n\twith sigma=10, rho=28, beta=8/3')
model.print()
'''





'''
    van der pol 
'''
def ode1(y, t, param1):
    return y[1] 
def ode2(y, t, param1):
    return -(param1 * (y[0] * y[0] - 1) * y[1] + y[0])



data = generate_lists(parse_csv('fake_vanderpol_data.csv'))

x, y, time = data
x = np.array(x)
y = np.array(y)
time = np.array(time)

# x' = y
# y' = -(mu*(x*x - 1)*y - x)
X = np.stack((x, y), axis=-1)  # First column is x, second is y
poly_library = ps.PolynomialLibrary(degree=3)

model = ps.SINDy(feature_names=["x", "y"], feature_library=poly_library)
model.fit(X, t=time)
print('van der pol \n\tmu = 1.5')
model.print()


# print('actual\n', X)
X_model = model.simulate(X[0, :],time)
# print('model\n', X_model)
fig, ax = plt.subplots()
ax.plot(X[:, 0], X[:, 1], label='Data')
ax.plot(X_model[:, 0], X_model[:, 1], '--', label='model', color='r')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('van der pol')
ax.legend()
plt.show()