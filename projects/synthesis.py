import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d 
from utils import rk2_data

# data collectors



def generate_axis(time_data, syn_time ,data):
    
    new_data = []
    time = []
    
    
    for i in range(len(time_data)):
        time_data[i] = round(time_data[i], 5)
        syn_time[i] = round(syn_time[i], 5)
    
    
    for j in range(len(time_data)):
        # print('t = ', time_data[j])
        if time_data[j] in syn_time:
            new_data.append(data[j])
            time.append(time_data[j])
            #print(new_data)
    
    return (time, new_data)

'''
    Plotting the Collins' system of ODEs
'''

# differential equations
# params index
#  0 = aphla, 1 = beta, 2 = epsilon, 3 = gamma, 4 = delta 

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


def trim(time_data):
    
    min_t = min(time_data)
    max_t = max(time_data)
    
    new_list = []
    
    for t in time_data:
        if t > min_t and t < max_t:
            new_list.append(t)
            
    return new_list 

odes = [
    dx_1_dt, dx_2_dt, dx_3_dt, dx_4_dt,
    dy_1_dt, dy_2_dt, dy_3_dt, dy_4_dt
]
# params indecies
# 0 = aphla, 1 = beta, 2 = epsilon, 3 = gamma, 4 = delta 
params1 = [1, 2, .5, -.5, -.5]
params2 = [1, 2, .5,   1,   1]

###################################
#   initial values (pronk gate)
#     - (x1 = .8, x2 = -.5, x3 = 1.1, x4 = -.9)
#     - (y1 = .3, y2 = -.4, y3 = .7,  y4 = -.2)

condition1 = [.8, -.5, 1.1, -.9, .3, -.4,  .7, -.2]
condition2 = [ 0,   0,   0,   0,  0,   0,   0,   0]

data = rk2_data(condition1, 10000, odes, 0, 100, params1)

x_data = data[:4]
time_data = data[-1]

'''
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
# synthesizing data

time_data = np.array(time_data)


x1 = data[0]


# interpolating the data
f = interp1d(time_data, x1)

new_time = time_data 


new_x1 = f(new_time)

print('len(new_time) = ', len(new_time))
print('len(new_x1)   = ', len(new_x1))

# new_time * .075
syn_time = new_time * .075 + 3

syn_t, syn_x1 = generate_axis(new_time, syn_time, new_x1)





fig, ax = plt.subplots()


ax.plot(syn_t, syn_x1, label="synthesis")
ax.set_xlabel('t')
ax.set_ylabel('x1')
ax.set_title('synthesis')
plt.tight_layout()
plt.show()


