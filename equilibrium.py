import numpy as np
from numba import njit, float64
from scipy.optimize import root, brentq

R = 8.31446e-5 # m3·bar/(K·mol)

# Returns array of activity coefficients for a mixture at temperature T
@njit(cache=True)
def gamma_NRTL(T, x, A, B, alpha):
    x = np.asarray(x, dtype=float64)
    A = np.asarray(A, dtype=float64)
    B = np.asarray(B, dtype=float64)

    Tau = A + B/T
    G = np.exp(-alpha * Tau)

    # S[i] = sum_k(x[k] * G[k,i])
    # Used in the first term and two times on the second term of the equation
    S = x @ G

    # C[i] = sum_k(x[k] * Tau[k,i] * G[k,i]) / S[i]
    # First term of equation and also the negative term after tau (indexes are different but the value is the same)
    C = (x @ (Tau*G)) / S 

    ln_gamma = C + np.sum(
        (x * G / S) *
        (Tau - C),
        axis=1
    )
    return np.exp(ln_gamma)

@njit(cache=True)
def _calc_Bij(T, a, b, w, kij, Tc, Pc, Vc):
    aij = (a[:, None] + a[None, :]) / 2
    bij = (b[:, None] + b[None, :]) / 2
    wij = (w[:, None] + w[None, :]) / 2
    
    Tcij = (Tc[:, None]*Tc[None, :])**(1/2)*(1 - kij)
    Trij = T / Tcij
    
    q = Pc * Vc / Tc
    v = np.cbrt(Vc)
    
    Pcij = 4*Tcij * (q[:, None] + q[None, :]) / (v[:, None] + v[None, :])**3
    
    B0ij = 0.1445 - 0.330/Trij - 0.1385/Trij**2 - 0.0121/Trij**3 - 0.000607/Trij**8
    B1ij = 0.0637 + 0.331/Trij**2 - 0.423/Trij**3 - 0.008/Trij**8
    B2ij = aij/Trij**6 - bij/Trij**8
    return R*Tcij/Pcij * (B0ij + B1ij*wij + B2ij)

@njit(cache=True)
def phi_tsonopoulos (T, P, y, Bij):
    # def phi_tsonopoulos(T, P, y, a, b, w, kij, Tc, Pc, Vc):
    # Bij = _calc_Bij(T, a, b, w, kij, Tc, Pc, Vc)
    Bm = y @ Bij @ y
    return np.exp(P/(R*T) * (2*(Bij @ y) - Bm))

# Bubble point OF from Raoult's law:
# 0 = sum_i(x_i * gamma_i * P_i^sat(T)) - P
# Psats must be a function that returns a vector with Psat values of all components
def _bubble_OF_NRTL(T, P, x, fn_psats, nrtl_params):
    A, B, alpha = nrtl_params
    gammas = gamma_NRTL(T, x, A, B, alpha)
    return np.sum(x * gammas * fn_psats(T)) - P

# sol: solution, vector of either [T, x1, x2, ...] or [P, x1, x2, ...]
# fixed: either T or P, whatever not on sol
def _dew_OF_NRTL(sol, fixed, y, fn_psats, nrtl_params, solve_for='T'):
    y = np.asarray(y, dtype=float)
    x = np.r_[sol[1:], 1 - np.sum(sol[1:])]
    if solve_for == 'T':
        T, P = sol[0], fixed
    else:
        P, T = sol[0], fixed

    gammas = gamma_NRTL(T, x, *nrtl_params)
    Psats = fn_psats(T)
    return y*P - x*gammas*Psats


def _vle_OF_NRTL_virial(sol, fixed, z, fn_psats, nrtl_params, virial_params, solve_for='Tx'):
    P = fixed if solve_for[0] == 'T' else sol[0]
    T = fixed if solve_for[0] == 'P' else sol[0]

    sol_comp = np.r_[sol[1:], 1 - np.sum(sol[1:])]

    x = z if solve_for[1] == 'y' else sol_comp
    y = z if solve_for[1] == 'x' else sol_comp

    gammas = gamma_NRTL(T, x, *nrtl_params)
    Psats = fn_psats(T)
    Bij = _calc_Bij(T, *virial_params)
    phis = phi_tsonopoulos(T, P, y, Bij)
    phis_sat = np.exp(Psats * np.diag(Bij) / (R*T))

    return y*phis/phis_sat*P - x*Psats*gammas


def Ty_bub_NRTL(P, x, fn_psats, nrtl_params, T_bounds=(280, 400)):
    T = brentq(_bubble_OF_NRTL, T_bounds[0], T_bounds[1], args=(P, x, fn_psats, nrtl_params))
    gammas = gamma_NRTL(T, x, *nrtl_params)
    y = x * gammas * fn_psats(T) / P
    return T, y

def Py_bub_NRTL(T, x, fn_psats, nrtl_params):
    gammas = gamma_NRTL(T, x, *nrtl_params)
    Psats = fn_psats(T)
    P = np.sum(x * gammas * Psats)
    y = x * gammas * Psats / P
    return P, y

def Tx_dew_NRTL(P, y, fn_psats, nrtl_params, T_guess=350):
    y = np.asarray(y, dtype=float)
    guess = np.r_[T_guess, y[:-1]]
    sol = root(_dew_OF_NRTL, guess, args=(P, y, fn_psats, nrtl_params, 'T'))
    if not sol.success or np.max(np.abs(sol.fun)) > 1e-8:
        raise RuntimeError("Non converging")
    T = sol.x[0]
    x = np.r_[sol.x[1:], 1 - np.sum(sol.x[1:])]
    if T <= 0 or np.any(x < 0) or np.any(x > 1):
        raise ValueError("Non physical solution")
    return T, x

def Px_dew_NRTL(T, y, fn_psats, nrtl_params, P_guess=1):
    y = np.asarray(y, dtype=float)
    guess = np.r_[P_guess, y[:-1]]
    sol = root(_dew_OF_NRTL, guess, args=(T, y, fn_psats, nrtl_params, 'P'))
    if not sol.success or np.max(np.abs(sol.fun)) > 1e-8:
        raise RuntimeError("Non converging")
    P = sol.x[0]
    x = np.r_[sol.x[1:], 1 - np.sum(sol.x[1:])]
    if np.any(x < 0) or np.any(x > 1):
        raise ValueError("Non physical solution")
    return P, x


def Ty_bub_NRTL_virial(P, x, fn_psats, nrtl_params, virial_params, T_guess=350):
    x = np.asarray(x, dtype=float)
    guess = np.r_[T_guess, x[:-1]]

    sol = root(_vle_OF_NRTL_virial, guess, args=(P, x, fn_psats, nrtl_params, virial_params, 'Ty'))
    if not sol.success or np.max(np.abs(sol.fun)) > 1e-8:
        raise RuntimeError("Non converging")
    T = sol.x[0]
    y = np.r_[sol.x[1:], 1 - np.sum(sol.x[1:])]
    if T <= 0 or np.any(y < 0) or np.any(y > 1):
        raise ValueError("Non physical solution")
    return T, y

def Tx_dew_NRTL_virial(P, y, fn_psats, nrtl_params, virial_params, T_guess=350):
    y = np.asarray(y, dtype=float)
    guess = np.r_[T_guess, y[:-1]]
    
    sol = root(_vle_OF_NRTL_virial, guess, args=(P, y, fn_psats, nrtl_params, virial_params, 'Tx'))

    if not sol.success or np.max(np.abs(sol.fun)) > 1e-8:
            raise RuntimeError("Non converging")
    T = sol.x[0]
    x = np.r_[sol.x[1:], 1 - np.sum(sol.x[1:])]
    if T <= 0 or np.any(x < 0) or np.any(x > 1):
        raise ValueError("Non physical solution")
    return T, x

def Py_bub_NRTL_virial(T, x, fn_psats, nrtl_params, virial_params, P_guess=1.0):
    x = np.asarray(x, dtype=float)
    guess = np.r_[P_guess, x[:-1]]

    sol = root(_vle_OF_NRTL_virial, guess, args=(T, x, fn_psats, nrtl_params, virial_params, 'Py'))

    if not sol.success or np.max(np.abs(sol.fun)) > 1e-8:
        raise RuntimeError("Non converging")
    P = sol.x[0]
    y = np.r_[sol.x[1:], 1 - np.sum(sol.x[1:])]
    if P <= 0 or np.any(y < 0) or np.any(y > 1):
        raise ValueError("Non physical solution")
    return P, y
    

def Px_dew_NRTL_virial(T, y, fn_psats, nrtl_params, virial_params, P_guess=1.0):
    y = np.asarray(y, dtype=float)
    guess = np.r_[P_guess, y[:-1]]

    sol = root(_vle_OF_NRTL_virial, guess, args=(T, y, fn_psats, nrtl_params, virial_params, 'Px'))
    
    if not sol.success or np.max(np.abs(sol.fun)) > 1e-8:
        raise RuntimeError("Non converging")
    P = sol.x[0]
    x = np.r_[sol.x[1:], 1 - np.sum(sol.x[1:])]
    if P <= 0 or np.any(x < 0) or np.any(x > 1):
        raise ValueError("Non physical solution")
    return P, x


def Txy_data(P, fn_psats, nrtl_params, virial_params=None, num_points=100):
    x1 = np.linspace(1e-6, 1-1e-6, num_points)
    y1 = np.zeros(num_points)
    T = np.zeros(num_points)
    for i, xi in enumerate(x1):
        x = np.array([xi, 1-xi])
        if virial_params is None:
            Ti, yi = Ty_bub_NRTL(P, x, fn_psats, nrtl_params)
        else:
            Ti, yi = Ty_bub_NRTL_virial(P, x, fn_psats, nrtl_params, virial_params)
        T[i] = Ti
        y1[i] = yi[0]
    return T, x1, y1

def Pxy_data(T, fn_psats, nrtl_params, virial_params=None, num_points=100):
    x1 = np.linspace(1e-6, 1 - 1e-6, num_points)
    y1 = np.linspace(num_points)
    P = np.linspace(num_points)
    for i, yi in enumerate(x1):
        x = np.array([x1, 1-x1])
        if virial_params is None:
            Pi, yi = Py_bub_NRTL(T, x, fn_psats, nrtl_params)
        else:
            Pi, yi = Py_bub_NRTL_virial(T, x, fn_psats, nrtl_params, virial_params)
        P[i] = Pi
        y1[i] = yi[0]
    return P, x1, y1
