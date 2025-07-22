import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d 
from utils import rk2_data, generate_axis
import pysindy as ps
from sklearn.linear_model import Lasso
from odes import collins_odes

# data collectors


'''
    Plotting the Collins' system of ODEs
'''
# params indecies
# 0 = aphla, 1 = beta, 2 = epsilon, 3 = gamma, 4 = delta 
pronk_params = [1, 2, .5, -.5, -.5]
trot_params = [1, 2, .5,   1,   1]

###################################
#   initial values (pronk gate)
#     - (x1 = .8, x2 = -.5, x3 = 1.1, x4 = -.9)
#     - (y1 = .3, y2 = -.4, y3 = .7,  y4 = -.2)

initial_condition = [.8, -.5, 1.1, -.9, .3, -.4,  .7, -.2]
condition2 = [ 0,   0,   0,   0,  0,   0,   0,   0]

data = rk2_data(initial_condition, 10000, collins_odes, 0, 100, trot_params)

x_data = data[:4]
time_data = data[-1]

'''
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1)


plt.suptitle('Collins Trot Gait (Figure 15)')
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
x2 = data[1]
x3 = data[2]
x4 = data[3]


# interpolating the data
i = 1
for x in [x1, x2, x3, x4]:
    

    f = interp1d(time_data, x)

    new_time = time_data


    new_x = f(new_time)

    # print('len(new_time) = ', len(new_time))
    # print('len(new_x1)   = ', len(new_x))

    syn_time =  .12*new_time + 3.5

    syn_t, syn_x = generate_axis(new_time, syn_time, new_x)



    # pysindy
    X = np.stack((syn_x), axis=-1)
    # print('X\n', X)
    fourier_library = ps.FourierLibrary(n_frequencies=2, include_sin=True, include_cos=True)

    model = ps.SINDy(feature_names=['x' + str(i)], optimizer=Lasso(alpha=.1), feature_library=fourier_library)
    # print(syn_t)
    model.fit(X, t=np.array(syn_t))
    # print(X.shape)
    model.print()
    
    #print(X)
    # X_model = model.simulate(X, np.array(syn_t))
    
    fig, ax = plt.subplots()


    ax.scatter(syn_t, syn_x, label="synthesis")
    # ax.plot(syn_t, X[:, 0], color='r')
    ax.set_xlabel('t')
    ax.set_ylabel('x' + str(i))
    ax.set_title('synthesis')
    plt.tight_layout()
    plt.show()
    i += 1

