import numpy as np
from scipy.integrate import solve_ivp

def van_der_pol(t, x, mu=1.5):
    return [x[1], mu * (1 - x[0]**2) * x[1] - x[0]]

mu = 1.5
t_min, t_max, n_steps = 0, 20, 5000
t_eval = np.linspace(t_min, t_max, n_steps)
x0 = [2.0, 0.0]

sol = solve_ivp(van_der_pol, (t_min, t_max), x0, t_eval=t_eval, args=(mu,))
x1 = sol.y[0]
x2 = sol.y[1]
t = sol.t

with open("fake_vanderpol_data.csv", "w") as f:
    f.write("#x,y,t\n")
    for xi, yi, ti in zip(x1, x2, t):
        f.write(f"{xi},{yi},{ti}\n")

print("Created 'van_der_pol_5000.csv' with 5000 rows.")