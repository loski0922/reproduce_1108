import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import warnings

# --- 논문 스타일(흑백)을 위한 Matplotlib 전역 설정 ---
mpl.rcParams['font.family'] = 'serif'  # Times New Roman과 유사한 세리프 폰트 사용
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.labelsize'] = 10
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['axes.edgecolor'] = 'black'  # 축 테두리
mpl.rcParams['xtick.color'] = 'black'  # 축 눈금
mpl.rcParams['ytick.color'] = 'black'  # 축 눈금
mpl.rcParams['axes.labelcolor'] = 'black'  # 축 라벨
mpl.rcParams['text.color'] = 'black'  # 모든 텍스트
mpl.rcParams['grid.linestyle'] = ':'  # 그리드 스타일
mpl.rcParams['grid.alpha'] = 0.5  # 그리드 투명도


# --- 유틸리티 함수 (이전과 동일) ---

def db_to_linear(db):
    return 10 ** (db / 10)


def linear_to_db(linear):
    return 10 * np.log10(np.maximum(linear, 1e-10))


def linear_to_dbm(watts):
    return 10 * np.log10(np.maximum(watts, 1e-10) * 1000)


def dbm_to_watts(dbm):
    return 10 ** ((dbm - 30) / 10)


def generate_channels(K, M):
    R_matrices = []
    for _ in range(K):
        A = np.random.randn(M, M) + 1j * np.random.randn(M, M)
        R = A @ A.conj().T
        R_matrices.append(R)
    return R_matrices


def calculate_psi_matrix(U, R_matrices, K, M):
    Psi = np.zeros((K, K), dtype=complex)
    for k in range(K):
        for i in range(K):
            if k == i:
                Psi[k, i] = 0
            else:
                u_k = U[:, k]
                R_i = R_matrices[i]
                Psi[k, i] = u_k.conj().T @ R_i @ u_k
    return np.real(Psi)


def calculate_D_matrix(U, R_matrices, target_gammas, K):
    S_i = np.zeros(K)
    for i in range(K):
        u_i = U[:, i]
        R_i = R_matrices[i]
        S_i[i] = np.real(u_i.conj().T @ R_i @ u_i)

    D_diag = target_gammas / np.maximum(S_i, 1e-10)
    return np.diag(D_diag)


def compute_uplink_sinr(U, q, R_matrices, K):
    sinr_ul = np.zeros(K)
    for i in range(K):
        u_i = U[:, i]
        R_i = R_matrices[i]
        signal = q[i] * np.real(u_i.conj().T @ R_i @ u_i)
        interference = 0
        for k in range(K):
            if i == k:
                continue
            R_k = R_matrices[k]
            interference += q[k] * np.real(u_i.conj().T @ R_k @ u_i)
        noise = 1.0
        sinr_ul[i] = signal / (interference + noise)
    return sinr_ul


# --- 핵심 알고리즘 (이전과 동일) ---
def run_beamforming_algorithm(R_matrices, K, M, target_gammas, P_max_watts, mode='P2', max_iter=100, epsilon=1e-5):
    n = 0
    q = np.ones(K) * 0.1
    C_n = 0

    U = np.random.randn(M, K) + 1j * np.random.randn(M, K)
    U = U / np.linalg.norm(U, axis=0, keepdims=True)

    for n in range(1, max_iter + 1):
        q_prev = q.copy()

        U_n = np.zeros((M, K), dtype=complex)
        for i in range(K):
            Q_i = np.eye(M, dtype=complex)
            for k in range(K):
                if i == k:
                    continue
                Q_i += q_prev[k] * R_matrices[k]

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", scipy.linalg.LinAlgWarning)
                    eigvals, eigvecs = scipy.linalg.eig(R_matrices[i], Q_i)
                u_i_n = eigvecs[:, np.argmax(np.real(eigvals))]
            except scipy.linalg.LinAlgError:
                u_i_n = U[:, i]

            U_n[:, i] = u_i_n / np.linalg.norm(u_i_n)

        U = U_n

        # --- D, Psi 계산 (U^(n) 기반) ---
        R_array = np.array(R_matrices)
        S_i = np.einsum('ik,kij,jk->k', U.conj(), R_array, U)
        S_i = np.real(S_i)

        D = np.diag(target_gammas / np.maximum(S_i, 1e-10))

        Psi = np.zeros((K, K))
        for k in range(K):
            for i in range(K):
                if k != i:
                    Psi[k, i] = np.real(U[:, k].conj().T @ R_matrices[i] @ U[:, k])

        Psi_T = Psi.T

        is_feasible = (C_n >= 1.0)

        if mode == 'P1' or not is_feasible:
            ones_K = np.ones(K)
            D_sigma = D @ ones_K
            TL = D @ Psi_T
            TR = D_sigma
            BL = (1.0 / P_max_watts) * (ones_K.T @ TL)
            BR = (1.0 / P_max_watts) * (ones_K.T @ TR)

            Lambda = np.zeros((K + 1, K + 1))
            Lambda[:K, :K] = TL
            Lambda[:K, K] = TR
            Lambda[K, :K] = BL
            Lambda[K, K] = BR

            eigvals, eigvecs = np.linalg.eig(Lambda)
            lambda_max = np.max(np.real(eigvals))
            C_n = 1.0 / lambda_max

            q_ext = np.real(eigvecs[:, np.argmax(np.real(eigvals))])
            q = q_ext[:K] / np.maximum(q_ext[K], 1e-10)
            q = np.maximum(q, 0)

            q_sum = np.sum(q)
            if q_sum > P_max_watts:
                q = (q / q_sum) * P_max_watts
        else:
            try:
                I = np.eye(K)
                ones_K = np.ones(K)
                D_sigma = D @ ones_K
                q = np.linalg.inv(I - D @ Psi_T) @ D_sigma
                q = np.maximum(np.real(q), 0)
            except np.linalg.LinAlgError:
                C_n = 0
                q = q_prev
                continue

        sinr_ul_n = compute_uplink_sinr(U, q_prev, R_matrices, K)
        relative_sinr_inv = target_gammas / np.maximum(sinr_ul_n, 1e-10)
        balance_diff = np.max(relative_sinr_inv) - np.min(relative_sinr_inv)

        if balance_diff < epsilon:
            if mode == 'P1':
                q_sum = np.sum(q)
                if q_sum > P_max_watts:
                    q = (q / q_sum) * P_max_watts
                sinr_ul_final = compute_uplink_sinr(U, q, R_matrices, K)
                C_opt = np.min(sinr_ul_final / np.maximum(target_gammas, 1e-10))
                return C_opt
            elif mode == 'P2':
                if C_n >= 1.0:
                    try:
                        I = np.eye(K)
                        ones_K = np.ones(K)
                        D_sigma = D @ ones_K
                        p_opt = np.linalg.inv(I - D @ Psi) @ D_sigma
                        p_opt = np.maximum(np.real(p_opt), 0)
                        return np.sum(p_opt)
                    except np.linalg.LinAlgError:
                        return np.inf
                else:
                    return np.inf

    if mode == 'P1':
        return C_n
    else:
        return np.inf


# --- 비교 대상 알고리즘 (이전과 동일) ---

def run_conventional_beamformer(R_matrices, target_gammas, K, M):
    U_conv = np.zeros((M, K), dtype=complex)
    for i in range(K):
        eigvals, eigvecs = np.linalg.eig(R_matrices[i])
        U_conv[:, i] = eigvecs[:, np.argmax(np.real(eigvals))]
        U_conv[:, i] /= np.linalg.norm(U_conv[:, i])

    R_array = np.array(R_matrices)
    S_i = np.einsum('ik,kij,jk->k', U_conv.conj(), R_array, U_conv)
    S_i = np.real(S_i)
    D = np.diag(target_gammas / np.maximum(S_i, 1e-10))

    Psi = np.zeros((K, K))
    for k in range(K):
        for i in range(K):
            if k != i:
                Psi[k, i] = np.real(U_conv[:, k].conj().T @ R_matrices[i] @ U_conv[:, k])

    try:
        p_opt = np.linalg.inv(np.eye(K) - D @ Psi) @ (D @ np.ones(K))
        p_opt = np.maximum(np.real(p_opt), 0)
        return np.sum(p_opt)
    except np.linalg.LinAlgError:
        return np.inf


def run_single_antenna(R_matrices, target_gammas, K):
    g = np.array([np.real(R[0, 0]) for R in R_matrices])
    D = np.diag(target_gammas / np.maximum(g, 1e-10))
    Psi = np.zeros((K, K))
    for k in range(K):
        for i in range(K):
            if k != i:
                Psi[k, i] = g[i]

    try:
        p_opt = np.linalg.inv(np.eye(K) - D @ Psi) @ (D @ np.ones(K))
        p_opt = np.maximum(np.real(p_opt), 0)
        return np.sum(p_opt)
    except np.linalg.LinAlgError:
        return np.inf


# --- 플롯 재현 함수 (스타일링 업데이트) ---
def plot_fig_4():
    """Fig. 4: Power Minimization 재현 (스타일링 강화)"""
    print("--- Fig. 4 (Power Minimization) 시뮬레이션 시작 ---")

    K = 5
    M = 4
    target_sinr_db = np.arange(5.0, 11.1, 0.5)
    target_sinr_linear = db_to_linear(target_sinr_db)

    power_proposed_dbm = []
    power_conv_dbm = []
    power_single_dbm = []

    R_matrices = generate_channels(K, M)
    P_max_watts_fig4 = dbm_to_watts(40)

    for gamma_lin in target_sinr_linear:
        gammas = np.ones(K) * gamma_lin /10
        print(f"  Target SINR = {linear_to_db(gamma_lin):.1f} dB 계산 중...")

        P_sum_proposed = run_beamforming_algorithm(R_matrices, K, M, gammas, P_max_watts_fig4, mode='P2')
        power_proposed_dbm.append(linear_to_dbm(P_sum_proposed))

        #P_sum_conv = run_conventional_beamformer(R_matrices, gammas, K, M)
        #power_conv_dbm.append(linear_to_dbm(P_sum_conv))

        #R_single = [R[0:1, 0:1] for R in R_matrices]
        #P_sum_single = run_single_antenna(R_single, gammas, K)
        #power_single_dbm.append(linear_to_dbm(P_sum_single))

    print("시뮬레이션 완료. 플롯 생성 중...")

    plt.figure(figsize=(7, 5))  # 논문 비율에 맞게 크기 조절

    # --- 스타일링 업데이트 ---
    # 흑백(k), 마커, 선 스타일 지정
    #plt.plot(target_sinr_db, power_single_dbm, 'v-k', label='single antenna', markersize=5)

    #plt.plot(target_sinr_db, power_conv_dbm, '+--k', label='conventional beamformer', markersize=7, linewidth=1)

    plt.plot(target_sinr_db, power_proposed_dbm, 'o-k', label='proposed algorithm', markersize=5, mfc='white', markeredgecolor='black')  # mfc='white': 속이 빈 원

    # 축 라벨 설정
    plt.xlabel('target SINR [dB] (identical for all users)')
    plt.ylabel('total transmission power [dBm]')

    # 축 범위와 눈금 설정 (이미지와 동일하게)
    plt.ylim(-6, 14)
    plt.xlim(4.8, 11.2)
    plt.xticks(np.arange(5, 12, 1))
    plt.yticks(np.arange(-5, 15, 5))

    # 범례 대신 텍스트 추가 (이미지와 유사하게)
    # plt.legend() # 범례 상자 대신 텍스트 사용
    #plt.text(6.5, 9.5, 'single antenna', ha='center', va='bottom')
    #plt.text(8.0, 1.0, 'conventional beamformer', ha='center', va='bottom')
    plt.text(9.0, -3.0, 'proposed algorithm', ha='center', va='bottom')

    # M, K 텍스트 추가
    plt.text(5.5, 10.0, 'M=4\nK=5', ha='center')

    plt.grid(True)  # 전역 설정(':', 0.5)에 따라 그리드 표시
    plt.tight_layout()  # 그림 저장 시 잘림 방지
    plt.show()


def plot_fig_5():
    """Fig. 5: Achievable SINR Margin 재현 (스타일링 강화, Y축 반전)"""
    print("\n--- Fig. 5 (SINR Balancing) 시뮬레이션 시작 ---")
    print("(이 시뮬레이션은 몬테카를로 횟수에 따라 수 분이 소요될 수 있습니다...)")

    M = 4
    K_values = np.arange(1, 16, 2)
    Pmax_dbm_values = np.arange(-40, 31, 10)
    Pmax_watts_values = dbm_to_watts(Pmax_dbm_values)

    MONTE_CARLO_RUNS = 20

    results_db = np.zeros((len(K_values), len(Pmax_dbm_values)))

    for i, K in enumerate(K_values):
        for j, P_max_watts in enumerate(Pmax_watts_values):
            print(f"  K = {K}, P_max = {Pmax_dbm_values[j]} dBm 계산 중...")

            C_opt_sum_linear = 0
            target_gammas_p1 = np.ones(K)*2
            runs_completed = 0

            for mc in range(MONTE_CARLO_RUNS):
                R_matrices = generate_channels(K, M)
                C_opt_linear = run_beamforming_algorithm(R_matrices, K, M,
                                                         target_gammas_p1,
                                                         P_max_watts, mode='P1')
                if np.isfinite(C_opt_linear):
                    C_opt_sum_linear += C_opt_linear
                    runs_completed += 1

            if runs_completed > 0:
                C_opt_avg_linear = C_opt_sum_linear / runs_completed
                results_db[i, j] = linear_to_db(C_opt_avg_linear)
            else:
                results_db[i, j] = -np.inf

    print("시뮬레이션 완료. 3D 플롯 생성 중...")

    # 3D Plot
    X, Y = np.meshgrid(Pmax_dbm_values, K_values)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    # --- 스타일링 업데이트 ---
    # 1. K (user) 별 라인 플롯 (P_max 변화) - 마커('o')와 선('-') 모두
    for i in range(len(K_values)):
        ax.plot(X[i, :], Y[i, :], results_db[i, :], 'o-k',
                markersize=4,
                mfc='white',  # 속이 빈 원
                markeredgecolor='black',
                linewidth=1)

    # 2. P_max 별 라인 플롯 (K 변화) - 선('-')만
    for j in range(len(Pmax_dbm_values)):
        ax.plot(X[:, j], Y[:, j], results_db[:, j], '-k',
                linewidth=1)

    # 라벨 (여백 'labelpad' 추가, Z축에 LaTeX 사용)
    ax.set_xlabel('total transmit power [dBm]', labelpad=10)
    ax.set_ylabel('number of users', labelpad=10)
    # r'' 문자열을 사용해 LaTeX의 \g(amma)를 인식시킴
    ax.set_zlabel(r'optimally balanced SINR$_i$ / $\gamma_i$ [dB]', labelpad=10)

    # 축 눈금 (이미지와 동일하게)
    ax.set_xticks(np.arange(-40, 31, 10))
    ax.set_yticks(K_values)
    ax.set_zticks(np.arange(-60, 41, 20))

    # --- Y축 반전 (사용자 요청) ---
    ax.invert_yaxis()
    # --------------------------

    # 뷰 각도 (이미지와 유사하게)
    ax.view_init(elev=20, azim=-120)

    # 배경 그리드 패널 스타일링
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill =False
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_edgecolor('black')

    # 3D 그리드 (전역 설정에 따라 ':')
    ax.grid(True)

    plt.tight_layout()
    plt.show()


# --- 메인 실행 ---
if __name__ == "__main__":
    # 경고 메시지 단순화
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    plot_fig_4()
    plot_fig_5()