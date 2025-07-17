import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# sovlers
def forward_euler(y0, n, f, t0, tf):
    
    y = y0
    t = t0
    dt = (tf-t0)/n
    
    for i in range(n):
        y = y + dt * f(y,t)
    
    return y

def system_forward_euler(y0, n, f, t0, tf, param1):
    y = y0
    t = t0
    dt = (tf-t0)/n
    
    for i in range(len(f)):
        ode = f[i]
        for j in range(n):
            y[i] = y[i] + dt * ode(y[i],t, param1)
    
    return y

def rk2(y0, n, f, t0, tf):
    y = y0
    t = t0
    dt = (tf-t0)/n
    
    for i in range(n):
        a = y + .5 * dt * f(y,t)
        y = y + dt * f(a, t + .5 * dt)
    
    return y

def system_rk2(y0, n, f, t0, tf, param1):
    
    y = y0
    t = t0
    dt = (tf-t0)/n
    
    for i in range(len(f)):
        ode = f[i]
        for j in range(n):
            a = y[i] + .5 * dt * ode(y[i],t, param1)
            y[i] = y[i] + dt * ode(a, t + .5 * dt, param1) 
    
    
    return y

# solutions
def actual_solution1(y0, t): # system
    
    y0 = np.array(y0)
    
    return y0 * np.exp(t)


# data collectors
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

# rhs functions
def rhs_function(vector, t, params, odes):
    new_vector = []
    for i in range(len(vector)):
        # print('ode = ', odes[i])
        new_vector.append(odes[i](vector, t, params))
        # print('new vector = ', new_vector)
    return new_vector 

# results

def fn(y, t, a):
    return a * y
y = [fn, fn]

'''
    FORWARD EULER testing suite for systems of ODEs
'''
# test 1
'''
print('\nTest 1')
approx_soln = system_forward_euler([4,2], 10000, y, 0, 10, 1)
actual_soln = actual_solution1([4,2], 10)

print(' Numerical Solution: ', approx_soln)
print(' Analytic Solution:  ', actual_soln, '\n')
'''
'''
# test 2
print('Test 2')
approx_soln = system_forward_euler([4,2], 100000, y, 0, 10, -3)
actual_soln = actual_solution1([4,2], -3*10)
print(' Numerical Solution: ', approx_soln)
print(' Analytic Solution:  ', actual_soln, '\n')

# test 3
print('Test 3')
approx_soln = system_forward_euler([7,10], 1000000, y, 0, 11, -2)
actual_soln = actual_solution1([7,10], -22)
print(' Numerical Solution: ', approx_soln)
print(' Analytic Solution:  ', actual_soln, '\n')
'''
'''
    RK2 testing suite for scalr first order ODES
'''
'''
# test 4
def ode1(y, t):
    return -y
    
def actual_solution2(y0, t): # scalar
    return y0*np.exp(-t)

print('Test 4')
approx_soln = rk2(4, 1000, ode1, 0, 5)
actual_soln = actual_solution2(4, 5)
print(' Numerical Solution (rk2):   ', approx_soln)
approx_soln = forward_euler(4, 1000, ode1, 0, 5)
print(' Numerical Solution (euler): ', approx_soln)
print(' Analytic Solution:  ', actual_soln, '\n')

# test 5
# y0 = 1, t0 = 0
def ode2(y, t):
    return -y + np.exp(-t)
    
def actual_solution3(y0, t): # scalar
    return np.exp(-t) * (1 + t)

print('Test 5')
approx_soln = rk2(1, 10000, ode2, 0, 0)
actual_soln = actual_solution3(1, 0)
print(' Numerical Solution: ', approx_soln)
print(' Analytic Solution:  ', actual_soln, '\n')
'''
'''
    RK2 testing suite for system of first order ODEs
'''
'''
print('Test 6')
approx_soln = system_rk2([4,2], 10000, y, 0, 10, 1)
actual_soln = actual_solution1([4,2], 10)
print(' Numerical Solution: ', approx_soln)
print(' Analytic Solution:  ', actual_soln, '\n')
'''

'''
    RK2 vs Forward-Euler for Van der pol oscillator
'''

def ode1(y, t, param1):
  return y[1] 
def ode2(y, t, param1):
   return -(param1 * (y[0] * y[0] - 1) * y[1] + y[0])

'''
# x = u, y = v
def ode3(u, v, param1):
    return v
def ode4(u, v, param1):
    return param1



print('Test 7')
euler_soln = system_forward_euler([.5, 0], 100, van_der_pol, 0, 30, 1.5)
rk2_soln = system_rk2([.5, 0], 100, van_der_pol, 0, 30, 1.5)

print(' Euler Solution: ', euler_soln)
print(' RK2 Solution:   ', rk2_soln, '\n')
'''

'''
    Ploting the van der pol oscillator using RK2
'''

van_der_pol = [ode1, ode2]
data = rk2_data([.5, 0], 1000, van_der_pol, 0, 30, 1.5)
print('Visualizing RK2 approximation')
# print(' rk2: ', system_rk2([.5, 0], 100, van_der_pol, 0, 30, 1.5))
print(' RK2 data: ', data)



# plotting solution
fig, ax = plt.subplots()
plt.title("van der pol approximation using RK2 Scheme")
plt.xlabel("x")
plt.ylabel("x'")
ax.scatter(data[0], data[1], color='b', label="IVP 1: (.5, 0)")
data = rk2_data([2.5, 1], 1000, van_der_pol, 0, 30, 1.5)
ax.scatter(data[0], data[1], color='r', label="IVP 2: (2.5, 1)")
data = rk2_data([-2, -1.5], 1000, van_der_pol, 0, 30, 1.5)
ax.scatter(data[0], data[1], color='g', label="IVP 3: (-2, -1.5)")
# ax.scatter(data[2], data[1], color='r', label="Numerical x'")
plt.legend()

plt.show()


'''
    Plotting the Collins' system of ODEs
'''

# differential equations
# params index
#  0 = aphla, 1 = beta, 2 = epsilon, 3 = gamma, 4 = delta 
'''
def dx_1_dt(vector, t, params):
    
    x1 = vector[0]
    x2 = vector[1]
    x4 = vector[3]
    y1 = vector[4]
    
    alpha = params[0]
    beta = params[1]
    epsilon = params[2]
    gamma = params[3]
    delta = params[4]
    
    equation = y1 - (alpha*x1*((x1**2)/3 - 1)) + beta + (epsilon*(x1**2)) + gamma*(x1 - x2) + delta*(x1 - x4)
     
    return  equation

def dx_2_dt(vector, t, params):
    
    x1 = vector[0]
    x2 = vector[1]
    x3 = vector[2]
    y2 = vector[5]

    alpha = params[0]
    beta = params[1]
    epsilon = params[2]
    gamma = params[3]
    delta = params[4]
    
    equation = y2 - (alpha*x2*((x2**2)/3 - 1)) + beta + (epsilon*(x2**2)) + gamma*(x2 - x1) + delta*(x2 - x3)
    
    return  equation

def dx_3_dt(vector, t, params):
    
    
    x2 = vector[1]
    x3 = vector[2]
    x4 = vector[3]
    y3 = vector[6]
    
    alpha = params[0]
    beta = params[1]
    epsilon = params[2]
    gamma = params[3]
    delta = params[4]
    
    equation = y3 - (alpha*x3*((x3**2)/3 - 1)) + beta + (epsilon*(x3**2)) + gamma*(x3 - x4) + delta*(x3 - x2)
     
    return  equation

def dx_4_dt(vector, t, params):
    
    x1 = vector[0]
    x3 = vector[2]
    x4 = vector[3]
    y4 = vector[7]
    
    alpha = params[0]
    beta = params[1]
    epsilon = params[2]
    gamma = params[3]
    delta = params[4]
    
    equation = y4 - (alpha*x4*((x4**2)/3 - 1)) + beta + (epsilon*(x4**2)) + gamma*(x4 - x3) + delta*(x4 - x1)
         
    return  equation

def dy_1_dt(vector, t, params):
    
    x1 = vector[0]
    x2 = vector[1]
    x3 = vector[2]
    x4 = vector[3]
    
    y1 = vector[4]
    y2 = vector[5]
    y3 = vector[6]
    y4 = vector[7]
    
    alpha = params[0]
    beta = params[1]
    epsilon = params[2]
    gamma = params[3]
    delta = params[4]
    
    equation = -x1
     
    
    return  equation

def dy_2_dt(vector, t, params):
    
    x1 = vector[0]
    x2 = vector[1]
    x3 = vector[2]
    x4 = vector[3]
    
    y1 = vector[4]
    y2 = vector[5]
    y3 = vector[6]
    y4 = vector[7]
    
    alpha = params[0]
    beta = params[1]
    epsilon = params[2]
    gamma = params[3]
    delta = params[4]
    
    equation = -x2
     
    
    return  equation

def dy_3_dt(vector, t, params):
    
    x1 = vector[0]
    x2 = vector[1]
    x3 = vector[2]
    x4 = vector[3]
    
    y1 = vector[4]
    y2 = vector[5]
    y3 = vector[6]
    y4 = vector[7]
    
    alpha = params[0]
    beta = params[1]
    epsilon = params[2]
    gamma = params[3]
    delta = params[4]
    
    equation = -x3
     
    
    return  equation

def dy_4_dt(vector, t, params):
    
    x1 = vector[0]
    x2 = vector[1]
    x3 = vector[2]
    x4 = vector[3]
    
    y1 = vector[4]
    y2 = vector[5]
    y3 = vector[6]
    y4 = vector[7]
    
    alpha = params[0]
    beta = params[1]
    epsilon = params[2]
    gamma = params[3]
    delta = params[4]
    
    equation = -x4
     
    
    return  equation

   

odes = [
    dx_1_dt, dx_2_dt, dx_3_dt, dx_4_dt,
    dy_1_dt, dy_2_dt, dy_3_dt, dy_4_dt
]
# params indecies
# 0 = aphla, 1 = beta, 2 = epsilon, 3 = gamma, 4 = delta 
params = [1, 2, .5, -.5, -.5]


###################################
#   initial values (pronk gate)
#     - (x1 = .8, x2 = -.5, x3 = 1.1, x4 = -.9)
#     - (y1 = .3, y2 = -.4, y3 = .7,  y4 = -.2)


condition1 = [.8, -.5, 1.1, -.9, .3, -.4,  .7, -.2]
condition2 = [ 0,   0,   0,   0,  0,   0,   0,   0]
data = rk2_data(condition1, 10000, odes, 0, 100, params)

x_data = data[:4]
time_data = data[-1]

# print('x_data: \n', x_data)
# print('time data: \n', time_data)


fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1)

ax1.plot(time_data, x_data[0], label="x1 solution", color='r')
ax1.set_xlabel('time t')
ax1.set_ylabel('x1(t)')

ax2.plot(time_data, x_data[1], label="x2 solution", color='b')
ax2.set_xlabel('time t')
ax2.set_ylabel('x2(t)')


ax3.plot(time_data, x_data[2], label="x3 solution", color='g')
ax3.set_xlabel('time t')
ax3.set_ylabel('x3(t)')

ax4.plot(time_data, x_data[3], label="x4 solution", color='black')
ax4.set_xlabel('time t')
ax4.set_ylabel('x4(t)')

ax1.legend()
ax2.legend()
ax3.legend()
ax4.legend()

plt.tight_layout()
plt.show()
'''

'''
    Lorenz System
'''

#                       RK2
# odes
# params[0] = sigma, params[1] = rho, params = beta
# y[0] = x, y[1] = y, y[2] = z
def ode1(vector, t, params):
    
    sigma, rho, beta = params
    x, y, z = vector
    
    return sigma*(y - x)

def ode2(vector, t, params):
    
    sigma, rho, beta = params
    x, y, z = vector
    
    return x*(rho - z) - y

def ode3(vector, t, params):
    
    sigma, rho, beta = params
    x, y, z = vector
    
    return x*y - beta*z

lorenz = [ode1, ode2, ode3]

params = [10, 28, 8/3]
condition = [0, 1, 20]
condition = [-8, 8, 27]
chart_data = rk2_data(condition, 10000, lorenz, 0, 30, params)
print('chart_data = ', chart_data[:3])
print('chart_data = ', chart_data[:3])
print('chart_data = ', chart_data[:3])

# Plot
ax = plt.figure().add_subplot(projection='3d')
ax.plot(chart_data[0], chart_data[1], chart_data[2])
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor")

plt.show()

