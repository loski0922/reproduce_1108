# reproduce_schubert_boche_2004.py
# Schubert & Boche (2004) - Solution of the Multiuser Downlink Beamforming Problem...
# Implements Table I (SINR balancing, P1) and Table II (Power minimization, P2),
# and plots figures analogous to Fig.1–6 in the paper.
# Reference for formulas/algorithmic structure: see equations (4)–(6), (12)–(16), (18)–(41).  [paper]

import numpy as np
import numpy.linalg as LA
from scipy.linalg import eigh, solve
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict

# -----------------------
# Utility helpers
# -----------------------

def unit_norm(v: np.ndarray) -> np.ndarray:
    n = LA.norm(v)
    return v / (n + 1e-16)

def herm(x: np.ndarray) -> np.ndarray:
    return np.conjugate(x.T)

def diagv(v: np.ndarray) -> np.ndarray:
    return np.diag(v)

def rand_unitary_vec(n: int) -> np.ndarray:
    z = (np.random.randn(n) + 1j*np.random.randn(n))/np.sqrt(2.0)
    return unit_norm(z)

def rand_spd(n: int) -> np.ndarray:
    """Random Hermitian positive semidefinite covariance."""
    M = (np.random.randn(n,n) + 1j*np.random.randn(n,n))/np.sqrt(2.0)
    R = M @ herm(M) / n
    # condition
    R += 1e-3 * np.eye(n)
    return R

# -----------------------
# Channel / covariance model
# -----------------------

def make_covariances(M: int, K: int, scenario: str = "iid") -> List[np.ndarray]:
    """
    Returns list of downlink covariance matrices R_k (M x M), k=1..K.
    - "iid": i.i.d. Rayleigh
    - "ULA": simple ULA one-ring style (optional)
    """
    Rs = []
    if scenario == "iid":
        for _ in range(K):
            Rs.append(rand_spd(M))
    else:
        # Simple ULA model with random AoAs
        d = 0.5  # lambda/2 spacing
        idx = np.arange(M)
        for _ in range(K):
            aoa = np.random.uniform(-np.pi/3, np.pi/3)
            a = np.exp(1j*2*np.pi*d*np.sin(aoa)*idx)
            a = a[:,None]
            R = a @ herm(a) + 0.1*rand_spd(M)
            Rs.append(R/np.trace(R)*M)
    return Rs

# -----------------------
# SINR and coupling matrices
# -----------------------

def gain(Ri: np.ndarray, u: np.ndarray) -> float:
    """u^H Ri u (real, >=0)"""
    return float(np.real(herm(u) @ Ri @ u))

def coupling_matrix_downlink(U: np.ndarray, Rs: List[np.ndarray]) -> np.ndarray:
    """
    C_DL(U): KxK, C[k,i] = (u_k^H R_i u_k)/(u_k^H R_k u_k).
    Diagonal entries are 1.
    """
    K = len(Rs)
    C = np.zeros((K,K), dtype=float)
    for k in range(K):
        gkk = gain(Rs[k], U[:,k])
        for i in range(K):
            C[k,i] = gain(Rs[i], U[:,k]) / (gkk + 1e-16)
    return C

def sigma_vec(U: np.ndarray, Rs: List[np.ndarray], sigma2: np.ndarray) -> np.ndarray:
    """
    sigma_k term normalized by desired-signal gain g_kk: s_k = sigma_k^2 / (u_k^H R_k u_k).
    """
    K = len(Rs)
    s = np.zeros(K, dtype=float)
    for k in range(K):
        s[k] = sigma2[k] / (gain(Rs[k], U[:,k]) + 1e-16)
    return s

def sinr_downlink(U: np.ndarray, p: np.ndarray, Rs: List[np.ndarray], sigma2: np.ndarray) -> np.ndarray:
    """
    SINR_k^DL = p_k g_kk / (sigma_k^2 + sum_{i≠k} p_i g_ki)
    """
    K = len(Rs)
    G = np.zeros((K,K), dtype=float)
    for k in range(K):
        for i in range(K):
            G[k,i] = gain(Rs[i], U[:,k])
    num = p * np.diag(G)
    den = sigma2 + (G @ p) - np.diag(G)*p
    return num / (den + 1e-16)

# -----------------------
# Extended coupling matrices (DL & UL)
# -----------------------

def A_extended_downlink(U: np.ndarray, Rs: List[np.ndarray],
                        gamma: np.ndarray, sigma2: np.ndarray, Ptot: float) -> np.ndarray:
    """
    A_DL = [[ D*C , D*sigma ], [ (1/P) 1^T , 0 ]]
    where D = diag(1/gamma), C = C_DL(U), sigma = sigma_vec(...).
    Eigen-system: A [p; 1] = (1/C*) [p; 1].
    """
    K = len(Rs)
    C = coupling_matrix_downlink(U, Rs)             # KxK
    D = np.diag(1.0/(gamma + 1e-16))                # KxK
    s = sigma_vec(U, Rs, sigma2)                    # K
    top_left  = D @ C
    top_right = (D @ s[:,None])                     # Kx1
    bot_left  = (1.0/Ptot) * np.ones((1,K))
    bot_right = np.zeros((1,1))
    A = np.block([[top_left,  top_right],
                  [bot_left,  bot_right]])
    return A

def A_extended_uplink(U: np.ndarray, Rs: List[np.ndarray],
                      gamma: np.ndarray, sigma2: np.ndarray, Ptot: float) -> np.ndarray:
    """
    For UL we use the transposed coupling in practice.
    A_UL = [[ D*C_UL , D*sigma ], [ (1/P) 1^T , 0 ]], with C_UL ≈ C_DL^T.
    """
    K = len(Rs)
    Cdl = coupling_matrix_downlink(U, Rs)
    Cul = Cdl.T
    D = np.diag(1.0/(gamma + 1e-16))
    s = sigma_vec(U, Rs, sigma2)
    top_left  = D @ Cul
    top_right = (D @ s[:,None])
    bot_left  = (1.0/Ptot) * np.ones((1,K))
    bot_right = np.zeros((1,1))
    A = np.block([[top_left,  top_right],
                  [bot_left,  bot_right]])
    return A

def dominant_eigvec(A: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Returns spectral radius and corresponding (right) eigenvector.
    (For nonnegative A we expect Perron eigenpair.)
    """
    vals, vecs = LA.eig(A)
    idx = np.argmax(np.real(vals))
    lam = np.real(vals[idx])
    v = vecs[:,idx]
    # enforce real-positive (up to scale) if possible
    if np.abs(v[-1]) < 1e-16:
        # avoid degenerate scaling; just return as-is
        return lam, v
    v = v / v[-1]   # normalize so that last component == 1  (paper eq. (13))
    v = np.real(v)
    return lam, v

# -----------------------
# Table I: SINR balancing (P1) - Alternating optimization
# -----------------------

def generalized_eigvec(Rsig: np.ndarray, Rint: np.ndarray) -> np.ndarray:
    """
    Dominant generalized eigenvector of (Rsig, Rint): solves Rint^{-1} Rsig u = lambda u.
    """
    # Stabilize
    Rint = (Rint + herm(Rint))/2
    Rsig = (Rsig + herm(Rsig))/2
    # Solve via eigh on Hermitian matrix
    w, V = eigh(Rsig, Rint, lower=False, check_finite=False)
    u = V[:, np.argmax(np.real(w))]
    return unit_norm(u)

def step_beamformers(U: np.ndarray, q: np.ndarray, Rs: List[np.ndarray], sigma2: np.ndarray) -> np.ndarray:
    """
    Corollary 2 / eq. (30): For given UL extended power q (first K comps), per-user beamformer = dominant generalized eigvec.
    """
    M = Rs[0].shape[0]
    K = len(Rs)
    Unew = np.zeros((M,K), dtype=complex)
    for k in range(K):
        Rint = (sigma2[k] + 0j) * np.eye(M, dtype=complex)
        for i in range(K):
            if i != k:
                Rint += q[i]*Rs[i]
        Unew[:,k] = generalized_eigvec(Rs[k], Rint)
    return Unew

def table1_p1(Rs: List[np.ndarray], sigma2: np.ndarray, gamma: np.ndarray,
              Ptot: float, maxit: int=50, tol: float=1e-4, verbose: bool=False) -> Dict:
    """
    Alternating optimization (Table I):
    - initialize q, U
    - repeat: update U (per-user generalized eigvec with current q),
              update q (dominant eigenvector of A_UL(U))
    - Stop when balanced margin C converges.
    Returns final U, p (DL power), q (UL power), C*, history.
    """
    M = Rs[0].shape[0]; K = len(Rs)
    # init
    U = np.column_stack([rand_unitary_vec(M) for _ in range(K)])
    q = np.ones(K) / K
    C_hist = []

    for it in range(maxit):
        # beamforming step (UL SINR maximization given q)
        U = step_beamformers(U, q, Rs, sigma2)

        # power-control step (UL): dominant eigenvector
        Aul = A_extended_uplink(U, Rs, gamma, sigma2, Ptot)
        rho_ul, v_ul = dominant_eigvec(Aul)   # v_ul = [q; 1]
        q = np.maximum(v_ul[:-1], 1e-12)  # positivity
        C = 1.0 / rho_ul
        C_hist.append(C)

        if verbose:
            print(f"[P1][it {it:02d}] C = {C:.6f}")

        if it > 1 and abs(C_hist[-1] - C_hist[-2]) < tol*np.maximum(1.0, abs(C_hist[-2])):
            break

    # Given final U, compute DL power p from DL extended matrix
    Adl = A_extended_downlink(U, Rs, gamma, sigma2, Ptot)
    rho_dl, v_dl = dominant_eigvec(Adl)   # v_dl = [p; 1]
    p = np.maximum(v_dl[:-1], 1e-12)
    Cstar = 1.0 / rho_dl

    return dict(U=U, p=p, q=q, C=Cstar, C_hist=C_hist)

# -----------------------
# Table II: Power minimization (P2)
# -----------------------

def solve_p_given_U_gamma(U: np.ndarray, Rs: List[np.ndarray],
                          gamma: np.ndarray, sigma2: np.ndarray) -> np.ndarray:
    """
    Equation set from SINR_k = gamma_k:
      p_k * g_kk = gamma_k * (sigma_k^2 + sum_{i≠k} p_i * g_ki)
    => sum_i M[k,i] p_i = gamma_k * sigma_k^2,
       where M[k,k]=g_kk, M[k,i]=-gamma_k*g_ki (i≠k).
    """
    K = len(Rs)
    G = np.zeros((K,K), dtype=float)
    for k in range(K):
        for i in range(K):
            G[k,i] = gain(Rs[i], U[:,k])

    M = np.array(G)
    for k in range(K):
        for i in range(K):
            if i != k:
                M[k,i] = -gamma[k] * G[k,i]
    # RHS
    b = gamma * sigma2
    p = solve(M, b, assume_a='gen')
    return np.maximum(p, 0.0)

def table2_p2(Rs: List[np.ndarray], sigma2: np.ndarray, gamma: np.ndarray,
              Ptot: float, maxit: int=50, tol: float=1e-4, verbose: bool=False) -> Dict:
    """
    Two-stage algorithm (Table II):
    1) Feasibility test via P1 (Table I) under total power constraint Ptot
    2) If feasible (C* >= 1), switch to power-min. step:
       - Repeat beamforming update (like Table I)
       - Power update by linear solve (41) to meet targets exactly
       - Stop when total power stops decreasing
    """
    # Stage 1: feasibility via P1
    out_p1 = table1_p1(Rs, sigma2, gamma, Ptot, maxit=maxit, tol=tol, verbose=verbose)
    Cstar = out_p1['C']
    if Cstar < 1.0:
        if verbose:
            print("[P2] Infeasible targets under given Ptot (C* < 1).")
        return dict(feasible=False, **out_p1)

    # Stage 2: minimize total power with equality SINR constraints
    U = out_p1['U']
    p = solve_p_given_U_gamma(U, Rs, gamma, sigma2)
    P_hist = [p.sum()]
    if verbose:
        print(f"[P2] Feasible. Init total power = {P_hist[-1]:.3f} (target satisfied).")

    for it in range(maxit):
        # Beamforming update for fixed virtual uplink q = p (heuristic linkage)
        q = p.copy()
        U = step_beamformers(U, q, Rs, sigma2)

        # Power update to meet SINR targets exactly
        p = solve_p_given_U_gamma(U, Rs, gamma, sigma2)
        P_hist.append(p.sum())

        if verbose:
            print(f"[P2][it {it:02d}] total power = {P_hist[-1]:.6f}")

        if it > 1 and abs(P_hist[-1] - P_hist[-2]) < tol*np.maximum(1.0, abs(P_hist[-2])):
            break

    return dict(feasible=True, U=U, p=p, P_hist=P_hist, C=Cstar)

# -----------------------
# Plotting: Figures 1–6 (analogs)
# -----------------------

def fig1_monotonic_C_vs_P():
    """
    Fig.1 analog: C*(U,P) strictly increasing in P.
    """
    np.random.seed(0)
    M, K = 6, 6
    Rs = make_covariances(M, K, "ULA")
    sigma2 = (1e-3)*np.ones(K)
    gamma = (np.ones(K))  # equal targets; actual scaling irrelevant for monotonicity demo
    P_list = np.linspace(1, 50, 12)
    C_list = []
    for P in P_list:
        out = table1_p1(Rs, sigma2, gamma, P, maxit=30, tol=5e-4)
        C_list.append(out['C'])
    plt.figure(figsize=(5,3.3))
    plt.plot(P_list, 10*np.log10(np.array(C_list)), marker='o')
    plt.xlabel('total power P')
    plt.ylabel('balanced margin 10log10(C*) [dB]')
    plt.title('Fig.1 (analog) — C* vs total power (monotone)')
    plt.grid(True, ls=':')
    plt.tight_layout()
    plt.savefig('fig1.png', dpi=180)

def fig2_convergence_curve():
    """
    Fig.2 analog: convergence behavior of alternating algorithm.
    """
    np.random.seed(1)
    M, K = 8, 8
    Rs = make_covariances(M, K, "iid")
    sigma2 = (1e-3)*np.ones(K)
    gamma = np.ones(K)
    P = 30.0
    out = table1_p1(Rs, sigma2, gamma, P, maxit=50, tol=1e-6)
    C_hist = out['C_hist']
    plt.figure(figsize=(5,3.3))
    plt.plot(10*np.log10(np.array(C_hist)), marker='o')
    plt.xlabel('iteration')
    plt.ylabel('balanced margin 10log10(C) [dB]')
    plt.title('Fig.2 (analog) — convergence of C')
    plt.grid(True, ls=':')
    plt.tight_layout()
    plt.savefig('fig2.png', dpi=180)

def baseline_p_single_antenna(Rs: List[np.ndarray], sigma2: np.ndarray,
                              gamma: np.ndarray) -> float:
    """
    Single-antenna baseline: choose the best single element (M=1) notion.
    For comparison, approximate by beamforming = standard basis e1.
    """
    M = Rs[0].shape[0]
    K = len(Rs)
    U = np.zeros((M,K), dtype=complex)
    for k in range(K):
        U[:,k] = np.eye(M)[:,0]  # pick first antenna only
    p = solve_p_given_U_gamma(U, Rs, gamma, sigma2)
    return p.sum()

def baseline_p_matched_filter(Rs: List[np.ndarray], sigma2: np.ndarray,
                              gamma: np.ndarray) -> float:
    """
    Conventional beamformer (spatial matched filter): u_k = principal eigenvector of R_k.
    """
    M = Rs[0].shape[0]
    K = len(Rs)
    U = np.zeros((M,K), dtype=complex)
    for k in range(K):
        w, V = LA.eigh((Rs[k]+herm(Rs[k]))/2)
        U[:,k] = unit_norm(V[:, np.argmax(w)])
    p = solve_p_given_U_gamma(U, Rs, gamma, sigma2)
    return p.sum()

def fig3_schematic_placeholder():
    """
    Fig.3 in paper is a schematic; here we skip (no numeric).
    (Optional) Could draw a block diagram if needed.
    """
    pass

def safe_db(x, floor_db=-120.0):
    x = float(x)
    if not np.isfinite(x) or x <= 0:
        return floor_db
    return 10*np.log10(x)

def fig4_power_min_compare():
    """
    Fig.4 analog: Compare total power (proposed Table II) vs suboptimal techniques.
    """
    np.random.seed(2)
    M, K = 6, 6
    Rs = make_covariances(M, K, "ULA")
    sigma2 = (1e-3)*np.ones(K)
    gamma = (10**(5/10))*np.ones(K)   # e.g., target 5 dB per user
    Ptot = 100.0

    # Proposed (Table II)
    out = table2_p2(Rs, sigma2, gamma, Ptot, maxit=25, tol=1e-6)
    P_prop = out['p'].sum() if out['feasible'] else np.nan

    # Baselines
    P_sa = baseline_p_single_antenna(Rs, sigma2, gamma)
    P_mf = baseline_p_matched_filter(Rs, sigma2, gamma)

    plt.figure(figsize=(5,3.3))
    names = ['single-antenna','matched-filter','proposed']
    vals = [safe_db(P_sa), safe_db(P_mf), safe_db(P_prop)]
    plt.bar(names, vals)
    plt.ylabel('required total power [dB arbitrary]')
    plt.title('Fig.4 (analog) — power minimization comparison')
    plt.tight_layout()
    plt.savefig('fig4.png', dpi=180)

def fig5_margin_vs_users_and_power():
    """
    Fig.5 analog: Achievable C* vs number of users & total transmit power.
    3D wireframe similar to the paper.
    """
    np.random.seed(3)
    M = 8
    users_list = np.arange(1, 16, 2)   # 1..15 odd
    P_list = np.array([-40,-30,-20,-10,0,10,20,30], dtype=float)
    # Convert dBm-ish to linear arbitrary scale:
    Plin = 10**((P_list - P_list.min())/10.0)  # just monotonic mapping for demo

    Z = np.zeros((len(P_list), len(users_list)))
    for ip, P in enumerate(Plin):
        for iu, K in enumerate(users_list):
            Rs = make_covariances(M, K, "ULA")
            sigma2 = (1e-3)*np.ones(K)
            gamma = np.ones(K)
            out = table1_p1(Rs, sigma2, gamma, P, maxit=25, tol=5e-4)
            Z[ip, iu] = safe_db(out['C'])  # C가 <=0 또는 NaN이면 floor dB로 대체

    # 3D wireframe
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    X, Y = np.meshgrid(users_list, P_list)
    fig = plt.figure(figsize=(6,4.5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_wireframe(X, Y, Z)
    ax.set_xlabel('number of users')
    ax.set_ylabel('total transmit power [dBm]')
    ax.set_zlabel('optimally balanced SINR_i/γ_i [dB]')
    ax.set_title('Fig.5 (analog)')
    plt.tight_layout()
    plt.savefig('fig5.png', dpi=180)

def fig6_min_power_vs_users_and_target():
    """
    Fig.6 analog: Minimal required transmit power vs number of users and target SINR.
    """
    np.random.seed(4)
    M = 8
    users_list = np.arange(3, 16, 2)    # 3..15 odd
    target_dB = np.array([-10, -5, 0, 5, 10], dtype=float)
    reqP = np.zeros((len(target_dB), len(users_list)))

    for it, tdB in enumerate(target_dB):
        for iu, K in enumerate(users_list):
            Rs = make_covariances(M, K, "ULA")
            sigma2 = (1e-3)*np.ones(K)
            gamma = (10**(tdB/10.0))*np.ones(K)
            # Increase Ptot until feasible (bisection-like)
            P = 1.0
            for _ in range(15):
                out = table1_p1(Rs, sigma2, gamma, P, maxit=20, tol=1e-3)
                if out['C'] >= 1.0:
                    P *= 0.7  # try smaller
                else:
                    P *= 1.6  # need more
            # Refine around current P
            out = table1_p1(Rs, sigma2, gamma, P, maxit=25, tol=1e-3)
            reqP[it, iu] = 10*np.log10(out['p'].sum() + 1e-12)  # dB-like

    plt.figure(figsize=(6,4))
    for it, tdB in enumerate(target_dB):
        plt.plot(users_list, reqP[it,:], marker='o', label=f'γ={tdB:.0f} dB')
    plt.xlabel('number of users')
    plt.ylabel('minimal required total power [dB arbitrary]')
    plt.title('Fig.6 (analog)')
    plt.grid(True, ls=':')
    plt.legend()
    plt.tight_layout()
    plt.savefig('fig6.png', dpi=180)

# -----------------------
# Main
# -----------------------

if __name__ == "__main__":
    fig1_monotonic_C_vs_P()
    fig2_convergence_curve()
    fig4_power_min_compare()
    fig5_margin_vs_users_and_power()
    fig6_min_power_vs_users_and_target()
    print("Saved: fig1.png … fig6.png")
