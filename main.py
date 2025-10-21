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
    "beta1": 0.4e-9,
    #Parametri chemioterapici
    "alphaA": 0.5e6,
    "gammaA": 0.3e-9,
    "tau": 2.5
}

parameters["alphaN"] = 0.6 * parameters["alphaA"]
parameters["gammaN"] = 0.6 * parameters["gammaA"]
beta3_vals = [0.28e-9, 0.32e-9]

def compute_variables (beta3_val, p):
    plocal = p.copy()
    plocal ["beta3"] = beta3_val
    lA = plocal["rA"] - (plocal["miA"] + plocal["epsilonA"])
    beta1_th = (plocal["miN"] * plocal["rA"]) / (lA * plocal["KA"])
    beta3_th = plocal["miN"] / plocal["rN"] * lA
    a = plocal["beta1"] * plocal["rA"] / plocal["KA"]
    b = lA * (beta1_th - plocal["beta1"])
    c = plocal["rN"] * (plocal["beta3"] - beta3_th)
    Delta = b ** 2 - 4 * a * c
    if plocal["beta3"] > beta3_th:
        eta = (plocal["rA"] * plocal["rN"] * ((plocal["beta3"] - beta3_th)) / (plocal["KA"] * lA ** 2))
        beta1_th_delta = beta1_th + 2 * eta + 2 * np.sqrt(eta * (beta1_th + eta))  # da capire
    else:
        eta = np.nan
        beta1_th_delta = np.nan
    return a, b, c, beta1_th, beta3_th, Delta, eta, beta1_th_delta

def find_equilibria(beta3_val,p):
    plocal = p.copy()
    plocal["beta3"] = beta3_val
    equilibria = []
    A0 = 0.0
    N0 = plocal["rN"] / plocal["miN"]
    equilibria.append([N0, A0])
    a, b, c, beta1_th, beta3_th, Delta, eta, beta1_th_delta = compute_variables(beta3_val, plocal)
    if Delta < 0:
        print("Delta negativo")
    else:
        D = np.sqrt(max(Delta,0))
        A1 = (-b - D)/(2 * a)
        A2 = (-b + D)/(2 * a)
        for A in [A1, A2]:
            if np.isreal(A) and A > 0:
                N = plocal["rN"]/(plocal["miN"] + plocal["beta1"] * A)
                equilibria.append([float(N), float(np.real(A))])
    return equilibria

equilibrium_1 = find_equilibria (beta3_vals[0], parameters)
equilibrium_2 = find_equilibria (beta3_vals[1], parameters)
print ("Equilibrio N1,A1:", equilibrium_1)
print ("Equilibrio N2,A2:", equilibrium_2)

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

def simulatecase1 (beta3_val, p, y0):
    plocal = p.copy()
    plocal["beta3"] = beta3_val
    sol = solve_ivp (fun=lambda t, y:ODE1 (t, y, plocal), t_span=(0, 2000), y0=y0, method='LSODA')
    return sol

eps = 1e3
eq_stabile2, saddle2, saddlepoint2 = [], [], []

for (N_eq, A_eq) in equilibrium_2:
    plocal = parameters.copy()
    plocal["beta3"] = beta3_vals[1]
    J, vals, vects = Jacobian_equilibria(N_eq, A_eq, plocal)
    if np.all((np.real(vals)) < 0):
        eq_stabile2.append([N_eq,A_eq])
    else:
        saddle2.append([N_eq,A_eq])

eq_stabile1, saddle1, saddlepoint1 = [], [], []

for (N_eq, A_eq) in equilibrium_1:
    plocal = parameters.copy()
    plocal["beta3"] = beta3_vals[0]
    J,vals, vects = Jacobian_equilibria(N_eq, A_eq, plocal)
    if np.all((np.real(vals) < 0)):
        eq_stabile1.append([N_eq,A_eq])
    else:
        saddle1.append([N_eq,A_eq])

for beta3_val in beta3_vals:
    a, b, c, beta1_th, beta3_th, Delta, eta, beta1_th_delta = compute_variables(beta3_val, parameters)
    print("beta1_th_delta: ",beta1_th_delta)
    # print("beta1_th: ",beta1_th)
    print("beta3_th: ", beta3_th)
    # print ("Delta: ", Delta)
    # print ("a:", a)
    # print("b: ", b)
    # print("c:", c)
    print ("beta1:", parameters["beta1"])
    print ("beta3:", beta3_val)

initialconditions = [[0.0, 0.0],
    [0.0, 0.001e8],[0.2e8, 0.2e8],[0.5e8, 0.5e8],[0.82e8, 0.3e8],[0.83e8, 0.12e8],[1.0e8, 0.5e8],[0.1e8, 0.05e8],
    [0.1e8, 0.2e8],[0.15e8, 0.1e8],[0.25e8, 0.05e8],[0.3e8, 0.25e8],[0.35e8, 0.15e8],[0.4e8, 0.3e8],[0.45e8, 0.4e8],
    [0.55e8, 0.1e8],[0.6e8, 0.2e8],[0.65e8, 0.35e8],[0.7e8, 0.5e8],[0.75e8, 0.2e8],[0.78e8, 0.1e8],[0.8e8, 0.25e8],
    [0.85e8, 0.4e8],[0.9e8, 0.6e8],[0.95e8, 0.3e8],[0.98e8, 0.15e8],[1.0e8, 0.1e8]]

N0 = parameters["rN"]/parameters["miN"]
initialconditionsA = [[N0 - 0.6e8, 0.6e8], [N0 - 0.07e8, 0.07e8]]
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
    sol = simulatecase1(beta3_vals[1], parameters, y0)
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
    sol = simulatecase1(beta3_vals[0], parameters, y0)
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
for y0 in initialconditions:
    sol = simulatecase1(beta3_vals[1], parameters, y0)
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
for y0 in initialconditions:
    sol = simulatecase1(beta3_vals[0], parameters, y0)
    plt.plot(sol.t, sol.y[0]/1e8, color='orange')
    plt.plot(sol.t, sol.y[1]/1e8, color='green')
plt.scatter([], [], color='orange', label='Sane')
plt.scatter([], [], color='green', label='Cancer')
plt.legend()
plt.tight_layout()
plt.show()

rho = 10
T = 7
n = int(4)
tmax = 105

def delta_approx(t, t0, sigma=0.1):
    return np.exp(-(t-t0)**2/(2*sigma**2))/(sigma * np.sqrt(2*np.pi))

def v_pulse(t, n, T, rho):
    v = 0
    for i in range(1, n + 1):
        v += rho * delta_approx(t, i * T, sigma=0.1)
    return v

def ODE2(t, y, p):
    plocal = p.copy()

    N = y[0]
    A = y[1]
    D = y[2]

    v = v_pulse(t, n, T, rho)
    dNdt = plocal["rN"] - plocal["miN"] * N - plocal["beta1"] * N * A - plocal["alphaN"] * plocal["gammaN"] * N * D
    dAdt = plocal["rA"] * A * (1 - A / plocal["KA"]) - plocal["beta3"] * N * A - (plocal["miA"] + plocal["epsilonA"]) * A - plocal["alphaA"] * plocal ["gammaA"] * A * D
    dDdt = v - plocal["gammaN"] * N * D - plocal["gammaA"] * A * D - plocal["tau"] * D
    return [dNdt, dAdt, dDdt]
def find_equilibria_system2(p, beta3_val):
    """
    Calcola equilibri del sistema 2 (con D) usando fsolve.
    Restituisce lista di equilibri [N,A,D].
    """
    plocal = p.copy()
    plocal["beta3"] = beta3_val
    equilibria2 = []

    # Primo caso: D = 0 (riduce al sistema 1)
    eq1 = find_equilibria(beta3_val, plocal)
    for N, A in eq1:
        equilibria2.append([N, A, 0.0])

    # Secondo caso: D > 0, si prova a risolvere il sistema completo
    def eq_full(vars):
        N, A, D = vars
        rN, miN, beta1, alphaN, gammaN = plocal["rN"], plocal["miN"], plocal["beta1"], plocal["alphaN"], plocal["gammaN"]
        rA, KA, beta3, miA, epsilonA, alphaA, gammaA, tau = plocal["rA"], plocal["KA"], plocal["beta3"], plocal["miA"], plocal["epsilonA"], plocal["alphaA"], plocal["gammaA"], plocal["tau"]
        fN = rN - miN*N - beta1*N*A - alphaN*gammaN*N*D
        fA = rA*A*(1 - A/KA) - beta3*N*A - (miA + epsilonA)*A - alphaA*gammaA*A*D
        fD = - gammaN*N*D - gammaA*A*D - tau*D
        return [fN, fA, fD]

    # Prova diversi punti di partenza per trovare eventuali equilibri con D>0
    guesses = [
        [eq1[-1][0], eq1[-1][1], 0.1],
        [eq1[-1][0]*0.9, eq1[-1][1]*0.9, 0.1],
        [eq1[-1][0]*1.1, eq1[-1][1]*1.1, 0.1]
    ]
    for guess in guesses:
        sol, info, ier, msg = fsolve(eq_full, guess, full_output=True)
        if ier == 1 and sol[2] > 0:  # sol convergente e D>0
            # Evitiamo duplicati simili
            if not any(np.allclose(sol, e, rtol=1e-5, atol=1e-8) for e in equilibria2):
                equilibria2.append(sol.tolist())

    return equilibria2
equilibria2 = find_equilibria_system2(parameters, beta3_vals[1])
print("\nEquilibri sistema 2 (con D) per beta3 =", beta3_vals[1])
for eq in equilibria2:
    print("N = {:.2e}, A = {:.2e}, D = {:.2e}".format(eq[0], eq[1], eq[2]))
def simulatecase2 (beta3_val, p, y0):
    plocal = p.copy()
    plocal["beta3"] = beta3_val
    sol_t, sol_y = [],[]
    y = y0.copy()

    sol = solve_ivp(fun=lambda t, y:ODE2(t, y, plocal), y0 = y,
                    t_span=(0,tmax),
                    method ='LSODA')
    sol_t.extend(sol.t)
    sol_y.extend(sol.y.T)

    return np.array(sol_t), np.array(sol_y)

P2 = equilibrium_2[2]
N0 = P2[0]
A0 = P2[1]
D0 = 0.0
y0_ = [N0, A0, D0]


sol_t, sol_y = simulatecase2(beta3_vals[1], parameters, y0_)
fig, ax1 = plt.subplots(figsize=(10,4))
ax1.plot(sol_t, sol_y[:,0], color='orange', label='N')
ax1.plot(sol_t, sol_y[:,1], color='green', label='A')
ax1.set_xlabel('Time')
ax1.set_ylabel('Cells')
ax1.grid()
ax2 = ax1.twinx()
ax2.plot(sol_t, sol_y[:,2], color='blue', linestyle='--', label='D')
ax2.set_ylabel('Drug')
plt.title('Regime II')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.tight_layout()
plt.show()
