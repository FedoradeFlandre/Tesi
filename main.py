import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from numpy.linalg import eig
import matplotlib.pyplot as plt

parameters = {
#Parametri biologici
    "miN": 0.01,
    "rN": 1e6,
    "rA": 0.05,
    "KA": 0.75e8,
    "miA": 0.01,
    "epsilonA": 0.01,
    #"beta1": 0.4e-9,
    #Parametri chemioterapici
    "alphaA": 0.5e8,
    "gammaA": 0.3e-8,
    "tau": 2.5
}

parameters["alphaN"] = 0.6 * parameters["alphaA"]
parameters["gammaN"] = 0.6 * parameters["gammaA"]
beta3_vals = [0.28e-9, 0.32e-9]
beta1 = 0.4e-9

def compute_variables (beta3_val, beta1_val, p):
    plocal = p.copy()
    plocal ["beta3"] = beta3_val
    plocal ["beta1"] = beta1_val
    lA = plocal["rA"] - (plocal["miA"] + plocal["epsilonA"])
    beta1_th = (plocal["miN"] * plocal["rA"]) / (lA * plocal["KA"])
    beta3_th = plocal["miN"] / plocal["rN"] * lA
    a = plocal["beta1"] * plocal["rA"] / plocal["KA"]
    b = lA * (beta1_th - plocal["beta1"])
    c = plocal["rN"] * (plocal["beta3"] - beta3_th)
    Delta = b ** 2 - 4 * a * c
    if plocal["beta3"] > beta3_th:
        eta = (plocal["rA"] * plocal["rN"] * ((plocal["beta3"] - beta3_th)) / (plocal["KA"] * lA ** 2))
        beta1_th_delta = beta1_th + 2 * eta + 2 * np.sqrt(eta * (beta1_th + eta))
        beta3_th_delta = beta3_th + ((lA ** 2 * (plocal["beta1"] - beta1_th) ** 2)/(4 * plocal["beta1"] * (plocal["rA"]/plocal["KA"]) * plocal["rN"]))
    else:
        eta = np.nan
        beta1_th_delta = np.nan
        beta3_th_delta = np.nan
    return a, b, c, beta1_th, beta3_th, Delta, eta, beta1_th_delta, beta3_th_delta

def find_equilibria(beta3_val, beta1_val, p):
    plocal = p.copy()
    plocal["beta3"] = beta3_val
    plocal["beta1"] = beta1_val
    equilibria = []
    A0 = 0.0
    N0 = plocal["rN"] / plocal["miN"]
    equilibria.append([N0, A0])
    a, b, c, beta1_th, beta3_th, Delta, eta, beta1_th_delta, beta3_th_delta = compute_variables(beta3_val, beta1_val, plocal)
    if Delta < 0:
        #print("Delta negativo")
        pass
    else:
        D = np.sqrt(max(Delta,0))
        A1 = (-b - D)/(2 * a)
        A2 = (-b + D)/(2 * a)
        for A in [A1, A2]:
            if np.isreal(A) and A > 0:
                N = plocal["rN"]/(plocal["miN"] + plocal["beta1"] * A)
                equilibria.append([float(N), float(np.real(A))])
    return equilibria

equilibrium_1 = find_equilibria(beta3_vals[0], beta1, parameters)
equilibrium_2 = find_equilibria(beta3_vals[1], beta1, parameters)
print('Equilibrio 1:', equilibrium_1)
print('Equilibrio 2:', equilibrium_2)

def Jacobian_equilibria(N, A, p):
    plocal = p.copy()
    J11 = - plocal["miN"] - plocal["beta1"] * A
    J12 = - plocal["beta1"] * N
    J21 = - plocal["beta3"] * A
    J22 = plocal["rA"] * (1 - 2 * A/plocal["KA"]) - plocal["beta3"] * N - (plocal["miA"] + plocal["epsilonA"])
    J = np.array([[J11, J12], [J21, J22]])
    vals, vects = eig(J)
    return J, vals, vects

def ODE1(t,y,p):
    plocal = p.copy()
    N = y[0]
    A = y[1]
    dNdt = plocal["rN"] - plocal["miN"] * N - plocal["beta1"] * N * A
    dAdt = plocal["rA"] * A * (1 - A / plocal["KA"]) - plocal["beta3"] * N * A - (plocal["miA"] + plocal["epsilonA"]) * A
    return [dNdt, dAdt]

def simulatecase1 (beta3_val, beta1_val, p, y0, tmax):
    plocal = p.copy()
    plocal["beta3"] = beta3_val
    plocal["beta1"] = beta1_val #0.4e-9
    sol = solve_ivp(fun=lambda t, y:ODE1(t, y, plocal), t_span=(0, tmax), y0=y0, method='LSODA')
    return sol

eps = 1e3
eq_stabile2, saddle2, saddlepoint2 = [], [], []

for (N_eq, A_eq) in equilibrium_2:
    plocal = parameters.copy()
    plocal["beta3"] = beta3_vals[1]
    plocal["beta1"] = beta1
    J, vals, vects = Jacobian_equilibria(N_eq, A_eq, plocal)
    if np.all((np.real(vals)) < 0):
        eq_stabile2.append([N_eq,A_eq])
    else:
        saddle2.append([N_eq,A_eq])

eq_stabile1, saddle1, saddlepoint1 = [], [], []

for (N_eq, A_eq) in equilibrium_1:
    plocal = parameters.copy()
    plocal["beta3"] = beta3_vals[0]
    plocal["beta1"] = beta1
    J,vals, vects = Jacobian_equilibria(N_eq, A_eq, plocal)
    if np.all((np.real(vals) < 0)):
        eq_stabile1.append([N_eq,A_eq])
    else:
        saddle1.append([N_eq,A_eq])

print ('Eq stabile1:', eq_stabile1)
print ('Sella1:', saddle1)
print ('Eq stabile2:', eq_stabile2)
print ('Sella2:', saddle2)

for beta3_val in beta3_vals:
    a, b, c, beta1_th, beta3_th, Delta, eta, beta1_th_delta, beta3_th_delta = compute_variables(beta3_val, beta1 , parameters)
    print("beta1_th_delta: ",beta1_th_delta)
    print("beta3_th_delta: ",beta3_th_delta)
    print("beta1_th: ",beta1_th)
    print("beta3_th: ", beta3_th)
    print("Delta: ", Delta)
    print("a:", a)
    print("b: ", b)
    print("c:", c)
    print("beta1:", beta1)
    print("beta3:", beta3_val)

initialconditions = [[0.0, 0.0],
    [0.0, 0.001e8],[0.2e8, 0.2e8],[0.5e8, 0.5e8],[0.82e8, 0.3e8],[0.83e8, 0.12e8],[1.0e8, 0.5e8],[0.1e8, 0.05e8],
    [0.1e8, 0.2e8],[0.15e8, 0.1e8],[0.25e8, 0.05e8],[0.3e8, 0.25e8],[0.35e8, 0.15e8],[0.4e8, 0.3e8],[0.45e8, 0.4e8],
    [0.55e8, 0.1e8],[0.6e8, 0.2e8],[0.65e8, 0.35e8],[0.7e8, 0.5e8],[0.75e8, 0.2e8],[0.78e8, 0.1e8],[0.8e8, 0.25e8],
    [0.85e8, 0.4e8],[0.9e8, 0.6e8],[0.95e8, 0.3e8],[0.98e8, 0.15e8],[1.0e8, 0.1e8], [0.3e8, 0.7e8]]

N0 = parameters["rN"]/parameters["miN"]
initialconditionsA = [[N0 - 0.6e8, 0.6e8], [N0 - 0.07e8, 0.07e8], [N0 - 0.9e8, 0.9e8]]
initialconditionsB = [[N0 - 1, 1], [N0 - 0.07e8, 0.07e8]]
plt.figure(figsize=(10, 4))

plt.subplot(2, 2, 1)
plt.grid()
plt.xlabel('Normal cell/1e8')
plt.ylabel('Cancer cell/1e8')
plt.title('Regime II')

for (N_eq, A_eq) in equilibrium_2:
    stabile = any(
        np.isclose(Ns, N_eq, rtol=1e-5, atol=1e-8)
        and
        np.isclose(As, A_eq, rtol=1e-5, atol=1e-8)
        for (Ns, As) in eq_stabile2
    )
    if stabile:
        plt.scatter(N_eq/1e8, A_eq/1e8, marker='o', color='red')
    else:
        plt.scatter(N_eq/1e8, A_eq/1e8, marker='*', color='blue')

for y0 in initialconditions:
    sol = simulatecase1(beta3_vals[1], beta1, parameters, y0, tmax=2000)
    plt.plot(sol.y[0]/1e8, sol.y[1]/1e8, color='grey', alpha=0.3)
plt.scatter([], [], marker='o', color='red', label='Equilibrio stabile')
plt.scatter([], [], marker='*', color='blue', label='Equilibrio instabile')
plt.legend()

plt.subplot(2, 2, 2)
plt.grid()
plt.xlabel('Normal cell/1e8')
plt.ylabel('Cancer cell/1e8')
plt.title('Regime III')

for (N_eq,A_eq) in equilibrium_1:
    stabile= any(
        np.isclose(N_eq, Ns, rtol=1e-5, atol=1e-8)
        and
        (np.isclose(A_eq, As, rtol=1e-5, atol=1e-8))
        for (Ns, As) in eq_stabile1
    )
    if stabile:
        plt.scatter(N_eq/1e8, A_eq/1e8, marker='o', color='red')
    else:
        plt.scatter(N_eq/1e8, A_eq/1e8, marker='*', color='blue')

for y0 in initialconditions:
    sol = simulatecase1(beta3_vals[0], beta1, parameters, y0, tmax=2000)
    plt.plot(sol.y[0]/1e8, sol.y[1]/1e8, color='grey', alpha=0.3)
plt.scatter([], [], marker='o', color='red', label='Equilibrio stabile')
plt.scatter([], [], marker='*', color='blue', label='Equilibrio instabile')
plt.legend()
plt.tight_layout()

plt.subplot(2, 2, 3)
plt.grid()
plt.xlabel('Time')
plt.ylabel("Cells/1e8")
plt.title('Regime II')
#Facciamo una prova (con equilibrium 1 e 2 non va)
for y0 in initialconditionsA:
    sol = simulatecase1(beta3_vals[1], beta1, parameters, y0, tmax=8000)
    plt.plot(sol.t, sol.y[0]/1e8, color='orange')
    plt.plot(sol.t, sol.y[1]/1e8, color='green')
plt.scatter([], [], color='orange', label='Sane')
plt.scatter([], [], color='green', label='Cancer')
plt.legend()
plt.tight_layout()

plt.subplot(2, 2, 4)
plt.grid()
plt.xlabel('Time')
plt.ylabel("Cells/1e8")
plt.title('Regime III')
for y0 in initialconditionsB:
    sol = simulatecase1(beta3_vals[0], beta1, parameters, y0, tmax=8000)
    plt.plot(sol.t, sol.y[0]/1e8, color='orange')
    plt.plot(sol.t, sol.y[1]/1e8, color='green')
plt.scatter([], [], color='orange', label='Sane')
plt.scatter([], [], color='green', label='Cancer')
plt.legend()
plt.tight_layout()
plt.show()

rho = 10
T = 7
S = 365
doses = [4, 5, 6, 7]

def delta_approx(t, t0, sigma=0.5):
    return np.exp(-(t-t0)**2/(2*sigma**2))/(sigma * np.sqrt(2*np.pi))

def v_pulse(t, n, T, rho, sigma=0.5):
    v = 0.0
    for i in range(1, n + 1):
        center = i * T
        v += rho * delta_approx(t, center, sigma=sigma)
    return v

def u_pulse(t, n, T, rho, sigma=0.5, delay=365):
    u = 0.0
    for i in range(1, n+1):
        center = 5 * delay + i * T
        u += rho * delta_approx(t, center, sigma=sigma)
    return u

def ODE2(t, y, p, n, sigma=0.5):
    plocal = p.copy()

    N = y[0]
    A = y[1]
    D = y[2]

    v = v_pulse(t, n, T, rho, sigma=sigma)
    u = u_pulse(t, n, T, rho, sigma=sigma, delay=365)
    dNdt = plocal["rN"] - plocal["miN"] * N - plocal["beta1"] * N * A - plocal["alphaN"] * plocal["gammaN"] * N * D
    dAdt = plocal["rA"] * A * (1 - A / plocal["KA"]) - plocal["beta3"] * N * A - (plocal["miA"] + plocal["epsilonA"]) * A - plocal["alphaA"] * plocal ["gammaA"] * A * D
    dDdt = v + u - plocal["gammaN"] * N * D - plocal["gammaA"] * A * D - plocal["tau"] * D
    return [dNdt, dAdt, dDdt]

def simulatecase2 (beta3_val, beta1_val, p, y0, tmax, n, sigma=0.5):
    plocal = p.copy()
    plocal["beta3"] = beta3_val
    plocal["beta1"] = beta1_val
    y = y0.copy()
    pulse_times = np.array([i * T for i in range(1, n+1) if i * T <= tmax])
    base_eval = np.linspace(0, tmax, int(max(3000, tmax * 200)))
    t_eval=np.unique(np.concatenate([base_eval, pulse_times]))
    if pulse_times.size> 0:
        dt = np.diff(t_eval).min() if t_eval.size > 1 else 1e-3
        eps = max(dt * 0.2, min(0.01, dt))
        extra = np.unique(np.concatenate([pulse_times - eps, pulse_times + eps]))
        t_eval = np.unique(np.concatenate([t_eval, extra]))
    sol = solve_ivp(fun=lambda t, y:ODE2(t, y, plocal, n), y0=y,
                    t_span=(0, tmax),
                    t_eval=t_eval,
                    method='RK45',
                    rtol=1e-6, atol=1e-9)
    return sol.t, sol.y.T

D0 = 0.0

P2 = max(equilibrium_2, key=lambda x:x[1])
N0II = P2[0]
A0II = P2[1]

y0_II = [N0II, A0II, D0]

P3 = max(equilibrium_1, key=lambda x:x[1])
N0III = P3[0]
A0III = P3[1]

y0_III = [N0III, A0III, D0]
line_styles_dict = {4: '-', 5: '--', 6: ':', 7: '-.'}

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.grid()
plt.title('Regime II')
plt.xlabel('Time')
plt.ylabel('Cells')
for n in doses:
    sol_tII, sol_yII = simulatecase2(beta3_vals[1], beta1, parameters, y0_II, tmax=150, n=n, sigma=0.5)
    #print("min(D)=", sol_yII[2, :].min(), "max(D)=", sol_yII[2, :].max(), "num_points=", len(sol_tII))
    style = line_styles_dict.get(n, '-')
    plt.plot(sol_tII, sol_yII[:, 0] / 1e8, color='green', linestyle=style, label='N')
    plt.plot(sol_tII, 4 * sol_yII[:, 1] / 1e8, color='red', linestyle=style, label='A')
    plt.plot(sol_tII, sol_yII[:, 2] / 10, color='blue', linestyle=style, label='D')

plt.subplot(1, 2, 2)
plt.grid()
plt.title('Regime III')
plt.xlabel('Time')
plt.ylabel('Cells')
for n in doses:
    sol_tIII, sol_yIII = simulatecase2(beta3_vals[0], beta1, parameters, y0_III, tmax=150, n=n, sigma=0.5)
    style = line_styles_dict.get(n, '-')
    plt.plot(sol_tIII/7, sol_yIII[:, 0] / 1e8, color='green', linestyle=style)
    plt.plot(sol_tIII/7, 4 * sol_yIII[:, 1] / 1e8, color='red', linestyle=style)
    plt.plot(sol_tIII/7, sol_yIII[:, 2] / 10, color='blue', linestyle=style)
plt.scatter([], [], color='green', label='Sane')
plt.scatter([], [], color='red', label='Cancer')
plt.scatter([], [], color='blue', label='Drug')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.grid()
plt.title('Regime II')
plt.xlabel('Time(years)')
plt.ylabel('Cells')
for n in doses:
    sol_tII, sol_yII = simulatecase2(beta3_vals[1], beta1, parameters, y0_II, tmax=7300, n=n, sigma=0.5)
    style = line_styles_dict.get(n,'-')
    plt.plot(sol_tII/365, sol_yII[:, 0] / 1e8, color='green', linestyle=style)
    plt.plot(sol_tII/365, 4 * sol_yII[:, 1] / 1e8, color='red', linestyle=style)
    plt.plot(sol_tII/365, sol_yII[:, 2] / 10, color='blue', linestyle=style)
plt.subplot(1, 2, 2)
plt.grid()
plt.title('Regime III')
plt.xlabel('Time(years)')
plt.ylabel('Cells')
for n in doses:
    sol_tIII, sol_yIII = simulatecase2(beta3_vals[0], beta1, parameters, y0_III, tmax=7300, n=n, sigma=0.5)
    style = line_styles_dict.get(n,'-')
    plt.plot(sol_tIII/365, sol_yIII[:, 0] / 1e8, color='green', linestyle=style)
    plt.plot(sol_tIII/365, 4 * sol_yIII[:, 1] / 1e8, color='red', linestyle=style)
    plt.plot(sol_tIII/365, sol_yIII[:, 2] / 10, color='blue', linestyle=style)
plt.scatter([], [], color='green', label='Sane')
plt.scatter([], [], color='red', label='Cancer')
plt.scatter([], [], color='blue', label='Drug')
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

#Grafici di biforcazione
plt.figure(figsize=(8,5))

def compute_bifurcation_plot(param_range, p, fixed_param_value, varying='beta3', subplot_index=1):
    rami = {}
    for val in param_range:
        if varying == 'beta3':
            beta3_val = val
            beta1_val = fixed_param_value
        else:
            beta3_val = fixed_param_value
            beta1_val = val

        equilibria = find_equilibria(beta3_val, beta1_val, p)
        for i, eq in enumerate(equilibria):
            N_eq, A_eq = eq
            plocal= p.copy()
            plocal["beta3"] = beta3_val
            plocal["beta1"] = beta1_val
            J, vals, vect = Jacobian_equilibria(N_eq, A_eq, plocal)
            stabile = np.all(np.real(vals)< 0)
            rami.setdefault(i, []).append((val, A_eq, stabile))
            #if i not in rami:
             #   rami[i]=[]
            #rami[i].append((beta3_val, A_eq, stabile))

    color_list=['blue', 'red', 'green']
    A_means = [(i, np.mean(np.array(ramo)[:,1])) for i, ramo in rami.items()]

    #for i,ramo in rami.items():
     #   ramo_array = np.array(ramo)
     #   A_mean = np.mean(ramo_array[:,1])
     #   A_means.append((i, A_mean))
    A_means.sort(key=lambda x: x[1])

    color_map = {i_ramo: color_list[j % len(color_list)] for j, (i_ramo, _) in enumerate(A_means)}

    plt.subplot(2,2,subplot_index)
    for i, ramo in rami.items():
        ramo_array = np.array(ramo)
        x_vals = ramo_array[:,0]*1e9
        A_vals = ramo_array[:,1]/1e8
        stabile_mask=ramo_array[:,2].astype(bool)
        color=color_map[i]
        plt.plot(x_vals[stabile_mask], A_vals[stabile_mask], color=color, linestyle='-')
        plt.plot(x_vals[~stabile_mask], A_vals[~stabile_mask], color=color, linestyle='--')

    plt.xlabel(varying)
    plt.ylabel("A all'equilibrio")
    a, b, c, beta1_th, beta3_th, Delta, eta, beta1_th_delta, beta3_th_delta = compute_variables(beta3_val, beta1_val, p)
    if varying == 'beta3':
        plt.axvline(beta3_th*1e9, color='grey', linestyle='--', linewidth=2)
        plt.axvline(beta3_th_delta*1e9, color='grey', linestyle='--', linewidth=2)
    else:
        plt.axvline(beta1_th * 1e9, color='grey', linestyle='--', linewidth=2)
        plt.axvline(beta1_th_delta * 1e9, color='grey', linestyle='--', linewidth=2)

beta3_range = np.linspace(0.25e-9, 0.35e-9, 100)
compute_bifurcation_plot(beta3_range, parameters, fixed_param_value = 0.2e-9, varying='beta3', subplot_index=1)
compute_bifurcation_plot(beta3_range, parameters, fixed_param_value = 0.4e-9, varying='beta3', subplot_index=2)
beta1_range = np.linspace(0.15e-9, 0.45e-9, 100)
compute_bifurcation_plot(beta1_range, parameters, fixed_param_value=beta3_vals[0], varying="beta1", subplot_index=3)
compute_bifurcation_plot(beta1_range, parameters, fixed_param_value=beta3_vals[1], varying="beta1", subplot_index=4)
plt.scatter([], [], color='red', label='P2')
plt.scatter([], [], color='green', label='P1')
plt.scatter([], [], color='blue', label='P0')
plt.legend()
plt.show()