import numpy as np
from scipy.integrate import solve_ivp
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
initialconditions_orange = [[0.0, 0.0], [0.5e8, 0.5e8], [0.75e8, 0.2e8], [0.98e8, 0.15e8], [0.6e8,0.14e8], [0.14e8, 0.6e8]]
N0 = parameters["rN"]/parameters["miN"]
initialconditionsA = [[N0 - 0.6e8, 0.6e8], [N0 - 0.07e8, 0.07e8], [N0 - 0.9e8, 0.9e8]]
initialconditionsB = [[N0 - 1, 1], [N0 - 0.07e8, 0.07e8]]

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
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
for y0 in initialconditions_orange:
    sol = simulatecase1(beta3_vals[1], beta1, parameters, y0, tmax=2000)
    plt.plot(sol.y[0]/1e8, sol.y[1]/1e8, color='orange', alpha=0.3)
plt.scatter([], [], marker='o', color='red', label='Equilibrio stabile')
plt.scatter([], [], marker='*', color='blue', label='Equilibrio instabile')
plt.legend()

plt.subplot(1, 2, 2)
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
for y0 in initialconditions_orange:
    sol = simulatecase1(beta3_vals[1], beta1, parameters, y0, tmax=2000)
    plt.plot(sol.y[0]/1e8, sol.y[1]/1e8, color='orange', alpha=0.3)
plt.scatter([], [], marker='o', color='red', label='Equilibrio stabile')
plt.scatter([], [], marker='*', color='blue', label='Equilibrio instabile')
plt.legend()
plt.tight_layout()
plt.savefig("grafico_1.png", bbox_inches='tight')
plt.show()

plt.figure(figsize=(10,6))
plt.subplot(1, 2, 1)
plt.grid()
plt.xlabel('Time')
plt.ylabel("Cells/1e8")
plt.title('Regime II')

for y0 in initialconditionsA:
    sol = simulatecase1(beta3_vals[1], beta1, parameters, y0, tmax=8000)
    plt.plot(sol.t, sol.y[0]/1e8, color='orange')
    plt.plot(sol.t, sol.y[1]/1e8, color='green')
plt.scatter([], [], color='orange', label='Sane')
plt.scatter([], [], color='green', label='Cancer')
plt.legend()
plt.tight_layout()

plt.subplot(1, 2, 2)
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
        center = 1 * delay + i * T
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
    dDdt = v - plocal["gammaN"] * N * D - plocal["gammaA"] * A * D - plocal["tau"] * D
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
        dt= np.diff(t_eval).min() if t_eval.size>1 else 1e-3
        eps = max(dt*0.2, min(0.01,dt))
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
plt.title("Regime II")
plt.xlabel('Time(weeks)')
plt.ylabel('Cells')
for n in doses:
    sol_tII, sol_yII = simulatecase2(beta3_vals[1], beta1, parameters, y0_II, tmax=150, n=n, sigma=0.5)
    style = line_styles_dict.get(n, '-')
    plt.plot(sol_tII/7, sol_yII[:, 0] / 1e8, color='green', linestyle=style, label='N')
    plt.plot(sol_tII/7, 4 * sol_yII[:, 1] / 1e8, color='red', linestyle=style, label='A')
    plt.plot(sol_tII/7, sol_yII[:, 2] / 10, color='blue', linestyle=style, label='D')

plt.subplot(1, 2, 2)
plt.grid()
plt.title("Regime II")
plt.xlabel('Time(years)')
plt.ylabel('Cells')
for n in doses:
    sol_tII, sol_yII = simulatecase2(beta3_vals[1], beta1, parameters, y0_II, tmax=7300, n=n, sigma=0.5)
    style = line_styles_dict.get(n, '-')
    plt.plot(sol_tII / 365, sol_yII[:, 0] / 1e8, color='green', linestyle=style)
    plt.plot(sol_tII / 365, 4 * sol_yII[:, 1] / 1e8, color='red', linestyle=style)
    plt.plot(sol_tII / 365, sol_yII[:, 2] / 10, color='blue', linestyle=style)
plt.scatter([], [], color='green', label='Sane')
plt.scatter([], [], color='red', label='Cancer')
plt.scatter([], [], color='blue', label='Drug')

plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("grafico_2.png", bbox_inches='tight')
plt.show()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.grid()
plt.title("Regime III")
plt.xlabel('Time(weeks)')
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

plt.subplot(1,2,2)
plt.title("Regime III")
plt.grid()
plt.xlabel('Time(years)')
plt.ylabel('Cells')
for n in doses:
    sol_tIII, sol_yIII = simulatecase2(beta3_vals[0], beta1, parameters, y0_III, tmax=1825, n=n, sigma=0.5)
    style = line_styles_dict.get(n,'-')
    plt.plot(sol_tIII/365, sol_yIII[:, 0] / 1e8, color='green', linestyle=style)
    plt.plot(sol_tIII/365, 4 * sol_yIII[:, 1] / 1e8, color='red', linestyle=style)
    plt.plot(sol_tIII/365, sol_yIII[:, 2] / 10, color='blue', linestyle=style)
plt.scatter([], [], color='green', label='Sane')
plt.scatter([], [], color='red', label='Cancer')
plt.scatter([], [], color='blue', label='Drug')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("grafico_3.png", bbox_inches='tight')
plt.show()

plt.figure(figsize=(12,5))
plt.title("Regime III")
plt.grid()
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
plt.savefig("grafico_4.png", bbox_inches='tight')
plt.show()

#Grafici di biforcazione
plt.figure(figsize=(8,5))

def compute_bifurcation_plot(param_range, p, fixed_param_value, varying='beta3', subplot_index=1):
    rami = {'blue' : [], 'green' :[], 'red': []}

    plt.subplot(2,2,subplot_index)

    for val in param_range:
        if varying == 'beta3':
            beta3_val = val
            beta1_val = fixed_param_value
        else:
            beta3_val = fixed_param_value
            beta1_val = val

        equilibria = find_equilibria(beta3_val, beta1_val, p)
        equilibria_sorted = sorted(equilibria, key = lambda x : x[1])
        l = len(equilibria)

        for i, eq in enumerate(equilibria_sorted):
            N_eq, A_eq = eq
            plocal= p.copy()
            plocal["beta3"] = beta3_val
            plocal["beta1"] = beta1_val
            J, vals, vect = Jacobian_equilibria(N_eq, A_eq, plocal)

            stabile = np.all(np.real(vals)< 0)
            if l == 1:
                rami['blue'].append((val, A_eq, stabile))
            elif l == 2:
                rami['blue'].append((val, equilibria_sorted[0][1],
                                     np.all(np.real(Jacobian_equilibria(*equilibria_sorted[0], plocal)[1]) < 0)))
                rami['red'].append((val, equilibria_sorted[1][1],
                                    np.all(np.real(Jacobian_equilibria(*equilibria_sorted[1], plocal)[1]) < 0)))
            else:
                rami['blue'].append((val, equilibria_sorted[0][1],
                                     np.all(np.real(Jacobian_equilibria(*equilibria_sorted[0], plocal)[1]) < 0)))
                rami['green'].append((val, equilibria_sorted[1][1],
                                      np.all(np.real(Jacobian_equilibria(*equilibria_sorted[1], plocal)[1]) < 0)))
                rami['red'].append((val, equilibria_sorted[2][1],
                                    np.all(np.real(Jacobian_equilibria(*equilibria_sorted[2], plocal)[1]) < 0)))

    for color, ramo in rami.items():
        x_vals = np.array([r[0] for r in ramo], dtype=float)*1e9
        A_vals = np.array([r[1] for r in ramo], dtype=float)/1e8
        stabile_mask=np.array([r[2] for r in ramo], dtype= bool)

        plt.plot(x_vals[stabile_mask], A_vals[stabile_mask], color=color, linestyle='-')
        plt.plot(x_vals[~stabile_mask], A_vals[~stabile_mask], color=color, linestyle='--')

    plt.xlabel(varying)
    plt.ylabel("A all'equilibrio")
    a, b, c, beta1_th, beta3_th, Delta, eta, beta1_th_delta, beta3_th_delta = compute_variables(beta3_val, beta1_val, p)
    if varying == 'beta3':
        plt.axvline(beta3_th*1e9, color='grey', linestyle='--', linewidth=2)
        plt.text(beta3_th*1e9,0, r'$\beta_3^{th}$')
        if not np.isnan(beta3_th_delta):
            plt.axvline(beta3_th_delta*1e9, color='grey', linestyle='--', linewidth=2)
            plt.text(beta3_th_delta*1e9,0, r'$\beta_{3,\Delta}^{th}$')
    else:
        plt.axvline(beta1_th * 1e9, color='grey', linestyle='--', linewidth=2)
        plt.text(beta1_th * 1e9,0, r'$\beta_1^{th}$')
        if not np.isnan(beta1_th_delta):
            plt.axvline(beta1_th_delta * 1e9, color='grey', linestyle='--', linewidth=2)
            plt.text(beta1_th_delta * 1e9,0, r'$\beta_{1,\Delta}^{th}$')

beta3_range = np.linspace(0.25e-9, 0.35e-9, 1000)
compute_bifurcation_plot(beta3_range, parameters, fixed_param_value = 0.2e-9, varying='beta3', subplot_index=1)
plt.scatter([], [], color='red', label='P2')
plt.scatter([], [], color='green', label='P1')
plt.scatter([], [], color='blue', label='P0')
plt.legend()
compute_bifurcation_plot(beta3_range, parameters, fixed_param_value = 0.4e-9, varying='beta3', subplot_index=2)
plt.scatter([], [], color='red', label='P2')
plt.scatter([], [], color='green', label='P1')
plt.scatter([], [], color='blue', label='P0')
plt.legend()
beta1_range = np.linspace(0.15e-9, 0.45e-9, 1000)
compute_bifurcation_plot(beta1_range, parameters, fixed_param_value=beta3_vals[0], varying="beta1", subplot_index=3)
plt.scatter([], [], color='red', label='P2')
plt.scatter([], [], color='green', label='P1')
plt.scatter([], [], color='blue', label='P0')
plt.legend()
compute_bifurcation_plot(beta1_range, parameters, fixed_param_value=beta3_vals[1], varying="beta1", subplot_index=4)
plt.scatter([], [], color='red', label='P2')
plt.scatter([], [], color='green', label='P1')
plt.scatter([], [], color='blue', label='P0')
plt.legend()
plt.tight_layout(pad=3.0) 
plt.savefig("grafico_5.png", bbox_inches='tight')
plt.show()

plt.figure(figsize=(10,5))

beta1_curve=[]
beta3_vec= np.linspace(0.0, 4.0e-9, 200)

for beta3_val in beta3_vec:
    a, b, c, beta1_th, beta3_th, Delta, eta, beta1_th_delta, beta3_th_delta = compute_variables(beta3_val=beta3_val,beta1_val=0.4e-9,p=parameters)
    beta1_curve.append(beta1_th_delta)

plt.plot(np.array(beta3_vec)*1e9, np.array(beta1_curve)*1e9, color='grey', linewidth=2, linestyle=':', label=r'SN$(P_1,P_2)$')

plt.axvline(beta3_th*1e9, color='grey', linestyle='--', label=r'$\beta_{1,th}$')
plt.axhline(beta1_th*1e9, color='grey', linestyle='-.', label=r'$\beta_{3,th}$')

plt.text(beta3_th*1e9+0.02, 0.01, r'$\beta_3=\beta_3^{th}$', fontsize=12, ha='left', va='bottom')
plt.text(0.05, beta1_th*1e9+0.05, r'$\beta_1=\beta_1^{th}$', fontsize=12, ha='left', va='bottom')

plt.text(0.2, 1 , f"III\n$P_2$ GAS", fontsize=14, ha='center', va='center', multialignment='center')
plt.text(0.6, 0.5, f"I\n$P_0$ GAS", fontsize=14, ha='center', va='center', multialignment='center')
plt.text(0.4, 1.5, f"II\n$P_0,P_2$ LAS", fontsize=14, ha='center', va='center', multialignment='center')

plt.annotate(f'TC($P_0,P_2$)', xy=(0.3, 0.1), xytext=(0.1,0.1), arrowprops=dict(arrowstyle='->', color='black'), fontsize=12)
plt.annotate(f'SN($P_1,P_2$)', xy=(0.8,1.9), xytext=(0.8, 1.5), arrowprops=dict(arrowstyle='->', color='black'), fontsize=12)
plt.annotate(f'TC($P_0,P_1$)', xy=(0.3,1.75), xytext=(0.15, 1.5), arrowprops=dict(arrowstyle='->', color='black'), fontsize=12)
plt.annotate(r'$\beta_1=\beta_{1,\Delta}^{th}$'+' o\n'+r'$\beta_3=\beta_{3,\Delta}^{th}$', xy = (0.7468,1.752), xytext=(0.6,1.75), arrowprops=dict(arrowstyle='->', color='black'), fontsize=12)
plt.xlabel("β₁ (x10⁻⁹)")
plt.ylabel("β₃ (x10⁻⁹)")
plt.xlim(0,1)
plt.ylim(0,2)
plt.savefig("grafico_6.png", bbox_inches='tight')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

paziente1=parameters.copy()
paziente1["epsilonA"] = 0.01
plt.figure(figsize=(10,5))
plt.subplot(2,3,1)
plt.grid()
plt.title("Paziente1")
plt.xlabel("Time(weeks)")
plt.ylabel("Cell/1e8")
sol_t1, sol_y1 = simulatecase2(beta3_vals[1],0.4e-9,paziente1, y0_II, tmax=105, n=8)
plt.plot(sol_t1/7, sol_y1[:, 0] / 1e8, color='green', linestyle='-')
plt.plot(sol_t1/7, 4 * sol_y1[:, 1] / 1e8, color='red', linestyle='-')
plt.plot(sol_t1/7, sol_y1[:, 2] / 10, color='blue', linestyle='-')

plt.subplot(2,3,4)
plt.grid()
plt.title("Paziente1")
plt.xlabel("Time(years)")
plt.ylabel("Cell/1e8")
sol_t1, sol_y1 = simulatecase2(beta3_vals[1],0.4e-9,paziente1, y0_II, tmax=7300, n=8)
plt.plot(sol_t1/365, sol_y1[:, 0] / 1e8, color='green', linestyle='-')
plt.plot(sol_t1/365, 4 * sol_y1[:, 1] / 1e8, color='red', linestyle='-')
plt.plot(sol_t1/365, sol_y1[:, 2] / 10, color='blue', linestyle='-')

paziente2=parameters.copy()
paziente2["epsilonA"] = 0.95*0.01
plt.subplot(2,3,2)
plt.grid()
plt.title("Paziente2")
plt.xlabel("Time(weeks)")
plt.ylabel("Cell/1e8")
sol_t2, sol_y2 = simulatecase2(beta3_vals[1],0.4e-9,paziente2, y0_II, tmax=105, n=8)
plt.plot(sol_t2/7, sol_y2[:, 0] / 1e8, color='green', linestyle='-')
plt.plot(sol_t2/7, 4 * sol_y2[:, 1] / 1e8, color='red', linestyle='-')
plt.plot(sol_t2/7, sol_y2[:, 2] / 10, color='blue', linestyle='-')
plt.subplot(2,3,5)
plt.grid()
plt.title("Paziente2")
plt.xlabel("Time(years)")
plt.ylabel("Cell/1e8")
sol_t2, sol_y2 = simulatecase2(beta3_vals[1],0.4e-9,paziente2, y0_II, tmax=7300, n=8)
plt.plot(sol_t2/365, sol_y2[:, 0] / 1e8, color='green', linestyle='-')
plt.plot(sol_t2/365, 4 * sol_y2[:, 1] / 1e8, color='red', linestyle='-')
plt.plot(sol_t2/365, sol_y2[:, 2] / 10, color='blue', linestyle='-')

paziente3=parameters.copy()
paziente3["epsilonA"] = 0.9*0.01
plt.subplot(2,3,3)
plt.grid()
plt.title("Paziente3")
plt.xlabel("Time(weeks)")
plt.ylabel("Cell/1e8")
sol_t3, sol_y3 = simulatecase2(beta3_vals[1],0.4e-9,paziente3, y0_II, tmax=105, n=8)
plt.plot(sol_t3/7, sol_y3[:, 0] / 1e8, color='green', linestyle='-')
plt.plot(sol_t3/7, 4 * sol_y3[:, 1] / 1e8, color='red', linestyle='-')
plt.plot(sol_t3/7, sol_y3[:, 2] / 10, color='blue', linestyle='-')
plt.subplot(2,3,6)
plt.grid()
plt.title("Paziente3")
plt.xlabel("Time(years)")
plt.ylabel("Cell/1e8")
sol_t3, sol_y3 = simulatecase2(beta3_vals[1],0.4e-9,paziente3, y0_II, tmax=7300, n=8)
plt.plot(sol_t3/365, sol_y3[:, 0] / 1e8, color='green', linestyle='-')
plt.plot(sol_t3/365, 4 * sol_y3[:, 1] / 1e8, color='red', linestyle='-')
plt.plot(sol_t3/365, sol_y3[:, 2] / 10, color='blue', linestyle='-')

plt.scatter([], [], color='green', label='Sane')
plt.scatter([], [], color='red', label='Cancer')
plt.scatter([], [], color='blue', label='Drug')
plt.legend(loc='upper right')
plt.tight_layout()
plt.savefig("grafico_7.png", bbox_inches='tight')
plt.show()

initialconditions_withchem =[[0.0, 0.0, 0.0],[0.98e8, 0.15e8, 0.0], [0.6e8,0.14e8, 0.0], [0.14e8, 0.6e8, 0.0]]

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.title("Regime II")
plt.grid()

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

for y0 in initialconditions_withchem:
    _, y=simulatecase2(beta3_vals[1], beta1, parameters, y0, tmax=2000, n=7, sigma=0.5)
    plt.plot(y[:,0]/1e8, y[:,1]/1e8, color='orange', alpha = 0.3)
plt.scatter([], [], marker='o', color='red', label='Equilibrio stabile')
plt.scatter([], [], marker='*', color='blue', label='Equilibrio instabile')
plt.legend(loc="upper right")

plt.subplot(1,2,2)
plt.title("Regime III")
plt.grid()
for (N_eq, A_eq) in equilibrium_1:
    stabile = any(
        np.isclose(Ns, N_eq, rtol=1e-5, atol=1e-8)
        and
        np.isclose(As, A_eq, rtol=1e-5, atol=1e-8)
        for (Ns, As) in eq_stabile1
    )
    if stabile:
        plt.scatter(N_eq/1e8, A_eq/1e8, marker='o', color='red')
    else:
        plt.scatter(N_eq/1e8, A_eq/1e8, marker='*', color='blue')

for y0 in initialconditions_withchem:
    _, y = simulatecase2(beta3_vals[0], beta1, parameters, y0, tmax=2000, n=7, sigma=0.5)
    plt.plot(y[:, 0] / 1e8, y[:, 1] / 1e8, color='orange', alpha=0.3)
plt.scatter([], [], marker='o', color='red', label='Equilibrio stabile')
plt.scatter([], [], marker='*', color='blue', label='Equilibrio instabile')
plt.legend(loc = "upper right")
plt.tight_layout()
plt.savefig("grafico_8.png", bbox_inches='tight')
plt.show()