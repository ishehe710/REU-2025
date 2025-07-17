import math as Math
import matplotlib.pyplot as plt
import numpy as np


# differential equations
def dydt(y, t):
    return -y


# ode solver
def euler(y0, n, f, t0, tf):
    
    y = y0
    t = t0
    dt = (tf-t0)/n
    print('dt = ', dt)
    
    for i in range(n):
        # print('y = ', y)
        y = y + dt * f(y,t)
        
    
    return y

def actual_solution(y0, t):
    return y0*Math.exp(-t)

def error_percent(actual, approx):
    absolute_error = actual - approx
    print('diff = ', absolute_error)
    relative_error = absolute_error / actual
    return relative_error * 100

# example
# y' = -y, y0 = 4, want y(5), t0 = 0

'''
print('Group 1:')
print('n = 1, y(5) = ', euler(4, 1, dydt, 0, 5), '')
print('n = 10, y(5) = ', euler(4, 10, dydt, 0, 5), '')
print('n = 100, y(5) = ', euler(4, 100, dydt, 0, 5), '')
print('n = 1000, y(5) = ', euler(4, 1000, dydt, 0, 5), '')

print('Group 2:')
print('n = 10000, y(5) = ', euler(4, 10000, dydt, 0, 5), '')
print('n = 100000, y(5) = ', euler(4, 100000, dydt, 0, 5), '')
print('n = 1000000, y(5) = ', euler(4, 1000000, dydt, 0, 5), '')
print('n = 10000000, y(5) = ', euler(4, 10000000, dydt, 0, 5), '')
'''

# comparing solutions
approx = euler(4, 10000, dydt, 0, 5)
actual = actual_solution(4, 5)
print('Numerical vs Analytic solutions: ')
print('Numerical: y(5) = ', approx, '')
print('Analytical: y(5) = ', actual)
print('Percent error: ', error_percent(actual, approx))

# plots
def euler_data(y0, n, f, t0, tf):
    
    # list data
    time_plot = []
    function_plot = []
    
    y = y0
    t = t0
    dt = (tf-t0)/n
    temp_t = t
    time_plot.append(t)
    function_plot.append(y)
    for i in range(n):
        
        y = y + dt * f(y,t)
        temp_t += dt
        time_plot.append(temp_t)
        function_plot.append(y)
    
    return (time_plot, function_plot)

data = euler_data(4, 100, dydt, 0, 5)
fig, ax = plt.subplots()
plt.title("ODE approximation: y' = -y , y0 = 4, t0 = 0, and finding y(5)")
plt.xlabel('time (t)')
plt.ylabel('f(y(t), t)')
# print(data)

# numerical
ax.scatter(data[0], data[1], color='b', label='Analytical')

# actual
x = np.linspace(0, 5)
y = np.exp(-x) * 4

ax.plot(x, y, color='black', label='Numerical Solution', linewidth=3.5)

plt.legend()

plt.show()