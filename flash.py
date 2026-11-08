import numpy as np
from scipy.optimize import approx_fprime
from equilibrium import gamma_NRTL

def of_torre(u, T, P, z, fn_psats, nrtl_params):
    x = np.r_[u[:-1], 1 - np.sum(u[:-1])]
    fvap = u[-1]
    gammas = gamma_NRTL(T, x, *nrtl_params)
    Ks = gammas*fn_psats(T)/P
    return x*(1 + fvap*(Ks - 1)) - z[:3]

# Check if solution is valid
def is_physical(u, z):
    x = np.r_[u[:-1], 1 - np.sum(u[:-1])]
    fvap = u[-1]
    if not (z[3] < fvap < 1):
        return False
    if np.any(x <= 0):
        return False
    y = (z[:3] - (1 - fvap) * x) / fvap
    return np.all(y >= 0)

def flash_newton_damped(F, u0, z, tol=1e-9, max_iter=100):
    u = np.array(u0, dtype=float)
    if not is_physical(u, z):
        raise ValueError("Invalid initial guess")
    for k in range(max_iter):
        f = F(u)
        if np.max(np.abs(f)) < tol:
            return u
        J = approx_fprime(u, F, epsilon=1e-8)
        delta = np.linalg.solve(J, -f)
        lam = 1
        while lam > 1e-10:
            u_step = u + lam*delta
            if is_physical(u_step, z):
                f_step = F(u_step)
                if np.linalg.norm(f_step) < np.linalg.norm(f):
                    break
            lam = lam*0.5
        else:
            raise RuntimeError("Damping failed")
        u = u_step
    raise RuntimeError("Non converging")

