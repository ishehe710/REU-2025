import numpy as np





# solver stuff
def rk2(y0, n, f, t0, tf):
    y = y0
    t = t0
    dt = (tf-t0)/n
    
    for i in range(n):
        a = y + .5 * dt * f(y,t)
        y = y + dt * f(a, t + .5 * dt)
    
    return y

'''
    parameters
        - y0 is the intial condition
            - type: array of scalars
        - n is the number of time steps
            - type: integer
        - f is a array of functions
            - type: array of functions
        - t0 is the inital time for the initial condition
            - type: scalar
        - tf is the final time 
            - type: scalar
        - params options paramters for the functions of f
            - type: array of scalars
'''
def rk2_data(y0, n, f, t0, tf, params):

    time_data = []
    chart_data = []
    
    # generate list per axis
    for i in range(len(f)):
        chart_data.append([])
    
    y = np.array(y0)
    t = t0
    
    # time step
    dt = (tf-t0)/n
    
    
    last_time = t
    for _ in range(n):

        # stage 1 of RK2 scheme
        stage_1 = np.add(y , np.multiply(.5 * dt , rhs_function(y, t, params, f)))
        
        # stage 2 of RK2 scheme
        y = np.add(y, np.multiply(dt , rhs_function(stage_1, t, params, f)))

        
        time_data.append(last_time)
        last_time += dt
        
        for j in range(len(chart_data)):
            chart_data[j].append(y[j])

    chart_data.append(time_data)
    return chart_data

def rhs_function(vector, t, params, odes):
    new_vector = []
    
    for i in range(len(vector)):
        new_vector.append(odes[i](vector, t, params))
    
    return new_vector 