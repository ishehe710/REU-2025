
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

collins_odes = [
    dx_1_dt,
    dx_2_dt,
    dx_3_dt,
    dx_4_dt,
    dy_1_dt,
    dy_2_dt,
    dy_3_dt,
    dy_4_dt
]