import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings

np.random.seed(0)

# --- 유틸리티 함수 ---
def db_to_linear(db):
    """dB 값을 선형 값으로 변환"""
    return 10 ** (db / 10)


def linear_to_db(linear):
    """선형 값을 dB 값으로 변환"""
    # 0 또는 음수 값에 대한 로그 오류 방지
    return 10 * np.log10(np.maximum(linear, 1e-10))


def linear_to_dbm(watts):
    """Watts를 dBm으로 변환"""
    return 10 * np.log10(np.maximum(watts, 1e-10) * 1000)


def dbm_to_watts(dbm):
    """dBm을 Watts로 변환"""
    return 10 ** ((dbm - 30) / 10)


def generate_channels(K, M):
    """
    K명의 사용자에 대한 M x M 채널 공분산 행렬(R_i)을 생성합니다.
    논문 의 "azimuth dispersion"을 모델링하기 위해
    단순히 h*h^H (rank-1) 대신 full-rank 랜덤 행렬을 생성합니다.

    R_i = A_i * A_i^H 형태 (Hermitian, positive semidefinite)
    """
    # 잡음 전력을 1로 정규화한다고 가정 (Table I/II, Line 3) [cite: 316, 440]
    # 따라서 R_i = R_tilde_i
    R_matrices = []
    for _ in range(K):
        # M x M 크기의 랜덤 복소 행렬 생성
        A = np.random.randn(M, M) + 1j * np.random.randn(M, M)
        # R = A * A^H (Hermitian, positive semidefinite 보장)
        R = A @ A.conj().T
        R_matrices.append(R)
    return R_matrices


def calculate_psi_matrix(U, R_matrices, K, M):
    """
    논문의 (4) 에 정의된 결합 행렬 Psi(U)를 계산합니다.
    [Psi]_{ki} = u_k^H * R_i * u_k (k != i)
    """
    Psi = np.zeros((K, K), dtype=complex)
    for k in range(K):
        for i in range(K):
            if k == i:
                Psi[k, i] = 0
            else:
                u_k = U[:, k]
                R_i = R_matrices[i]
                Psi[k, i] = u_k.conj().T @ R_i @ u_k
    print(Psi)
    return np.real(Psi)  # SINR은 실수


def calculate_D_matrix(U, R_matrices, target_gammas, K):
    """
    논문의 (177) 에 정의된 대각 행렬 D를 계산합니다.
    D = diag(gamma_i / (u_i^H * R_i * u_i))
    """
    S_i = np.zeros(K)
    for i in range(K):
        u_i = U[:, i]
        R_i = R_matrices[i]
        S_i[i] = np.real(u_i.conj().T @ R_i @ u_i)

    # 0으로 나누기 방지
    D_diag = target_gammas / np.maximum(S_i, 1e-10)
    return np.diag(D_diag)


def compute_uplink_sinr(U, q, R_matrices, K):
    """
    (21) [cite: 251] 및 (440)의 정지 조건에 사용될 가상 업링크 SINR을 계산합니다.
    SINR_i = (q_i * u_i^H * R_i * u_i) / (sum_{k!=i} q_k * u_i^H * R_k * u_i + 1)
    """
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

        noise = 1.0  # 정규화된 잡음 (Line 3, Table II)

        sinr_ul[i] = signal / (interference + noise)
    return sinr_ul


# --- 핵심 알고리즘 (Table I & II) ---
def run_beamforming_algorithm(R_matrices, K, M, target_gammas, P_max_watts, mode='P2', max_iter=100, epsilon=1e-5):
    """
    논문의 Table I  (P1, SINR Balancing)과
    Table II  (P2, Power Minimization)를 구현한 핵심 함수.

    Args:
        R_matrices (list): 채널 공분산 행렬 리스트 [R_1, ..., R_K]
        K (int): 사용자 수
        M (int): 안테나 수
        target_gammas (np.array): 타겟 SINR (선형)
        P_max_watts (float): 최대 총 전력 (1단계용)
        mode (str): 'P1' (Fig 5) 또는 'P2' (Fig 4)
        max_iter (int): 최대 반복 횟수
        epsilon (float): 수렴 정지 조건

    Returns:
        mode='P1': C_opt (최대 SINR 마진, 선형)
        mode='P2': P_sum_opt (최소 총 전력, Watts)
    """

    # 1. 초기화 (Line 1, Table II)
    n = 0
    q = np.ones(K) * 0.1  # 0으로 시작 방지
    C_n = 0

    # U (빔포머) 초기화: (M x K) 행렬, 각 열이 u_i
    # 랜덤 초기화
    U = np.random.randn(M, K) + 1j * np.random.randn(M, K)
    U = U / np.linalg.norm(U, axis=0, keepdims=True)

    # R_tilde = R (Line 2, 3)  (noise_sigma^2 = 1 가정)

    # 4. 반복 (Line 4, Table II)
    for n in range(1, max_iter + 1):
        q_prev = q.copy()

        # 6. u_i^(n) 계산 (Line 6, Table II)
        # u_i = v_max(R_i, Q_i(q^(n-1)))
        U_n = np.zeros((M, K), dtype=complex)
        for i in range(K):
            # Q_i 계산 (31)
            # Q_i = sum_{k!=i} q_k * R_k + I
            Q_i = np.eye(M, dtype=complex)  # 잡음 항 (I)
            for k in range(K):
                if i == k:
                    continue
                Q_i += q_prev[k] * R_matrices[k]

            # 일반화된 고유값 문제 풀이 (30)
            # R_i * v = lambda * Q_i * v
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", scipy.linalg.LinAlgWarning)
                    eigvals, eigvecs = scipy.linalg.eig(R_matrices[i], Q_i)

                # v_max: 최대 고유값에 해당하는 고유 벡터 [cite: 329]
                u_i_n = eigvecs[:, np.argmax(np.real(eigvals))]
            except scipy.linalg.LinAlgError:
                # Q_i가 특이행렬일 경우, 이전 값 사용
                u_i_n = U[:, i]

            # 7. 정규화 (Line 7, Table II)
            U_n[:, i] = u_i_n / np.linalg.norm(u_i_n)

        U = U_n  # U^(n) 업데이트

        # --- D, Psi 계산 (U^(n) 기반) ---
        # S_i = u_i^H * R_i * u_i
        # R_matrices 리스트를 (K, M, M) 크기의 3D NumPy 배열로 변환
        R_array = np.array(R_matrices)
        # u_k^H * R_k * u_k 연산을 모든 k에 대해 한번에 계산
        S_i = np.einsum('ik,kij,jk->k', U.conj(), R_array, U)
        S_i = np.real(S_i)

        # D = diag(gamma_i / S_i)
        D = np.diag(target_gammas / np.maximum(S_i, 1e-10))

        # Psi_ki = u_k^H * R_i * u_k
        Psi = np.zeros((K, K))
        for k in range(K):
            for i in range(K):
                if k != i:
                    Psi[k, i] = np.real(U[:, k].conj().T @ R_matrices[i] @ U[:, k])

        Psi_T = Psi.T

        # --- 전력 할당 (Table II, Line 8-14)  ---

        # P1 모드 (Fig 5)는 항상 1단계만 실행
        # P2 모드 (Fig 4)는 C_n < 1 (목표 달성 불가)일 때 1단계,
        #                   C_n >= 1 (목표 달성 가능)일 때 2단계 실행

        # P2 모드에서 C_n=1 이 타겟이므로, target_gammas를 사용.
        # P1 모드에서 C_n은 찾는 값이므로, target_gammas=1을 사용.

        is_feasible = (C_n >= 1.0)  # P2 모드용 플래그

        if mode == 'P1' or not is_feasible:
            # 8. 1단계: SINR Balancing (Line 9, Table II / Table I) [cite: 316, 440]

            # Lambda(U, P_max) 행렬 구축 (16)
            # sigma = 1 (벡터)
            ones_K = np.ones(K)
            D_sigma = D @ ones_K

            # Top-Left (K x K): D * Psi^T
            TL = D @ Psi_T
            # Top-Right (K x 1): D * sigma
            TR = D_sigma
            # Bottom-Left (1 x K): (1/P_max) * 1^T * D * Psi^T
            BL = (1.0 / P_max_watts) * (ones_K.T @ TL)
            # Bottom-Right (1 x 1): (1/P_max) * 1^T * D * sigma
            BR = (1.0 / P_max_watts) * (ones_K.T @ TR)

            # Lambda 행렬 (K+1 x K+1)
            Lambda = np.zeros((K + 1, K + 1))
            Lambda[:K, :K] = TL
            Lambda[:K, K] = TR
            Lambda[K, :K] = BL
            Lambda[K, K] = BR

            # 9. 고유값 문제 풀이 (Line 9, Table II)
            # Lambda * q_ext = lambda_max * q_ext
            eigvals, eigvecs = np.linalg.eig(Lambda)
            lambda_max = np.max(np.real(eigvals))

            # 10. C^(n) = 1 / lambda_max (Line 10, Table II)
            C_n = 1.0 / lambda_max

            # 다음 반복을 위한 q^(n) 계산
            q_ext = np.real(eigvecs[:, np.argmax(np.real(eigvals))])
            q = q_ext[:K] / np.maximum(q_ext[K], 1e-10)  # 마지막 요소로 정규화 [cite: 213]
            q = np.maximum(q, 0)  # 파워는 양수

            if mode == 'P1':
                # P1은 총 전력이 P_max를 초과하지 않도록 q를 스케일링해야 함
                q_sum = np.sum(q)
                if q_sum > P_max_watts:
                    q = (q / q_sum) * P_max_watts

        else:  # mode == 'P2' and is_feasible
            # 11. 2단계: Power Minimization (Line 12, Table II)
            # q^(n) = (I - D * Psi^T)^-1 * D * 1
            try:
                I = np.eye(K)
                ones_K = np.ones(K)
                D_sigma = D @ ones_K  # D*1

                q = np.linalg.inv(I - D @ Psi_T) @ D_sigma
                q = np.maximum(np.real(q), 0)  # 파워는 양수

            except np.linalg.LinAlgError:
                # 역행렬 계산 불가 (특이 행렬)
                # 1단계로 되돌아가서 C_n을 다시 계산
                C_n = 0  # feasible하지 않다고 표시
                q = q_prev  # 이전 q 유지
                continue  # 다음 반복으로

        # 15. 정지 조건 (Line 15, Table II / Corollary 3) [cite: 440, 357]
        # max(gamma/SINR) - min(gamma/SINR) < epsilon
        # q_prev (n-1)과 U (n) 사용
        sinr_ul_n = compute_uplink_sinr(U, q_prev, R_matrices, K)

        # 0으로 나누기 방지
        relative_sinr_inv = target_gammas / np.maximum(sinr_ul_n, 1e-10)

        balance_diff = np.max(relative_sinr_inv) - np.min(relative_sinr_inv)

        if balance_diff < epsilon:
            # 수렴됨
            if mode == 'P1':
                ##################################################################################################################################
                # P1은 C_n (SINR 마진)을 반환
                # 마지막 q가 P_max를 만족하는지 확인하고 C_n 재계산
                q_sum = np.sum(q)
                if q_sum > P_max_watts:
                    q = (q / q_sum) * P_max_watts

                # 수렴된 U, q로 최종 C_n 계산
                sinr_ul_final = compute_uplink_sinr(U, q, R_matrices, K)
                # C = min(SINR_i / gamma_i) [cite: 163]
                C_opt = np.min(sinr_ul_final / np.maximum(target_gammas, 1e-10))
                return C_opt
                # return C_n
                ####################################################################################################################################

            elif mode == 'P2':
                # P2는 C_n >= 1 (feasible)인지 확인
                if C_n >= 1.0:
                    # 16. 최종 다운링크 전력 계산 (Line 16, Table II)
                    # p_opt = (I - D * Psi)^-1 * D * 1
                    try:
                        I = np.eye(K)
                        ones_K = np.ones(K)
                        D_sigma = D @ ones_K

                        # 주의: p_opt는 Psi (Psi^T 아님) 사용
                        p_opt = np.linalg.inv(I - D @ Psi) @ D_sigma
                        p_opt = np.maximum(np.real(p_opt), 0)

                        # 총 전력 반환 (Line 13)
                        return np.sum(p_opt)
                    except np.linalg.LinAlgError:
                        return np.inf  # 계산 실패
                else:
                    # 수렴했지만 목표 SINR 달성 불가 (infeasible)
                    return np.inf

    # 최대 반복 도달
    if mode == 'P1':
        return C_n  # 현재까지의 C_n 반환
    else:
        return np.inf  # 수렴 실패


# --- 비교 대상(Baseline) 알고리즘 (Fig. 4용) ---

def run_conventional_beamformer(R_matrices, target_gammas, K, M):
    """
    "Conventional Beamformer (spatial matched filter)"
    가장 간단한 형태인 R_i의 주 고유 벡터로 u_i를 고정.
    그 후 (Line 16) 과 동일한 DL 전력 제어를 적용[cite: 463].
    """
    U_conv = np.zeros((M, K), dtype=complex)
    for i in range(K):
        eigvals, eigvecs = np.linalg.eig(R_matrices[i])
        U_conv[:, i] = eigvecs[:, np.argmax(np.real(eigvals))]
        U_conv[:, i] /= np.linalg.norm(U_conv[:, i])

    # (Line 16)과 동일한 p_opt 계산
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
    """
    "Single Antenna" (M=1)
    M=1일 때, R_i는 스칼라 채널 이득 g_i = |h_i|^2.
    R_matrices[i][0, 0]을 g_i로 사용.
    """
    g = np.array([np.real(R[0, 0]) for R in R_matrices])

    # M=1 일 때, u_i = 1 (스칼라).
    # S_i = u_i^H * R_i * u_i = g_i
    # Psi_ki = u_k^H * R_i * u_k = g_i
    D = np.diag(target_gammas / np.maximum(g, 1e-10))
    Psi = np.zeros((K, K))
    for k in range(K):
        for i in range(K):
            if k != i:
                Psi[k, i] = g[i]  # 간섭 채널 이득

    try:
        p_opt = np.linalg.inv(np.eye(K) - D @ Psi) @ (D @ np.ones(K))
        p_opt = np.maximum(np.real(p_opt), 0)
        return np.sum(p_opt)
    except np.linalg.LinAlgError:
        return np.inf


# --- 플롯 재현 함수 ---

def plot_fig_4():
    """Fig. 4: Power Minimization 재현"""
    print("--- Fig. 4 (Power Minimization) 시뮬레이션 시작 ---")

    K = 5  # 사용자 수 [cite: 426]
    M = 4  # 안테나 수 [cite: 426]

    target_sinr_db = np.arange(5.0, 11.1, 0.5)
    target_sinr_linear = db_to_linear(target_sinr_db)

    power_proposed_dbm = []
    power_conv_dbm = []
    power_single_dbm = []

    # "for a specific channel": 채널을 한 번만 생성
    R_matrices = generate_channels(K, M)

    # P_max는 1단계에서만 사용. 충분히 큰 값으로 설정.
    P_max_watts_fig4 = dbm_to_watts(40)  # 40 dBm = 10 Watts

    for gamma_lin in target_sinr_linear:
        gammas = np.ones(K) * gamma_lin
        print(f"  Target SINR = {linear_to_db(gamma_lin):.1f} dB 계산 중...")

        # 1. Proposed Algorithm (Table II)
        P_sum_proposed = run_beamforming_algorithm(R_matrices, K, M, gammas, P_max_watts_fig4, mode='P2')
        power_proposed_dbm.append(linear_to_dbm(P_sum_proposed))

        # 2. Conventional Beamformer
        P_sum_conv = run_conventional_beamformer(R_matrices, gammas, K, M)
        power_conv_dbm.append(linear_to_dbm(P_sum_conv))

        # 3. Single Antenna (M=1)
        # M=4 채널의 첫 번째 안테나 요소만 사용
        R_single = [R[0:1, 0:1] for R in R_matrices]
        # K=5, M=1로 간주하고 실행
        P_sum_single = run_single_antenna(R_single, gammas, K)
        power_single_dbm.append(linear_to_dbm(P_sum_single))

    print("시뮬레이션 완료. 플롯 생성 중...")

    plt.figure(figsize=(10, 6))
    plt.plot(target_sinr_db, power_single_dbm, 'v-', label='single antenna (M=1)', markersize=5)
    plt.plot(target_sinr_db, power_conv_dbm, '+--', label='conventional beamformer (MF)', markersize=8)
    plt.plot(target_sinr_db, power_proposed_dbm, 'o-', label='proposed algorithm (Table II)', markersize=5)

    plt.xlabel('target SINR [dB] (identical for all users)')
    plt.ylabel('total transmission power [dBm]')
    plt.title('Fig. 4 재현: Power Minimization Capability (K=5, M=4)')
    plt.legend()
    plt.grid(True)
    #plt.ylim(-6, 14)  # 원본 [cite: 422-429]과 유사하게 Y축 설정
    plt.show()


def plot_fig_5():
    """Fig. 5: Achievable SINR Margin (SINR Balancing) 재현"""
    print("\n--- Fig. 5 (SINR Balancing) 시뮬레이션 시작 ---")
    print("(이 시뮬레이션은 몬테카를로 횟수에 따라 수 분이 소요될 수 있습니다...)")

    M = 4  # 안테나 수 (Fig. 4에서 고정) [cite: 426]
    K_values = np.arange(1, 16, 2)  # 1, 3, 5, ..., 15 [cite: 457]
    Pmax_dbm_values = np.arange(-40, 31, 10)  # -40, -30, ..., 30 [cite: 459]
    Pmax_watts_values = dbm_to_watts(Pmax_dbm_values)
    # "randomly distributed"  -> 몬테카를로 시뮬레이션
    MONTE_CARLO_RUNS = 20  # 시간 관계상 횟수를 줄임 (논문 재현시 100~1000회 필요)

    results_db = np.zeros((len(K_values), len(Pmax_dbm_values)))

    for i, K in enumerate(K_values):
        for j, P_max_watts in enumerate(Pmax_watts_values):
            print(f"  K = {K}, P_max = {Pmax_dbm_values[j]} dBm 계산 중...")

            C_opt_sum_linear = 0
            # "equal targets gamma_1 = ... = gamma_K" [cite: 460]
            # C_opt = SINR_i / gamma_i 를 찾는 것이므로, gamma_i = 1 로 설정
            target_gammas_p1 = np.ones(K)

            runs_completed = 0
            for mc in range(MONTE_CARLO_RUNS):
                R_matrices = generate_channels(K, M)

                C_opt_linear = run_beamforming_algorithm(R_matrices, K, M, target_gammas_p1, P_max_watts, mode='P1')

                if np.isfinite(C_opt_linear):
                    C_opt_sum_linear += C_opt_linear
                    runs_completed += 1

            if runs_completed > 0:
                C_opt_avg_linear = C_opt_sum_linear / runs_completed
                results_db[i, j] = linear_to_db(C_opt_avg_linear)
            else:
                results_db[i, j] = -np.inf  # 계산 실패

    print("시뮬레이션 완료. 3D 플롯 생성 중...")
    # 3D Plot
    X, Y = np.meshgrid(Pmax_dbm_values, K_values)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # --- 원하는 플롯 구현: x–z 방향 선만 그리기 ---
    for i in range(len(K_values)):
        ax.plot(Pmax_dbm_values,  # x축 (P_max)
            np.full_like(Pmax_dbm_values, K_values[i]),  # y축 (고정)
            results_db[i, :],  # z축
            'o-', c='black')

    ax.set_xlabel('total transmit power [dBm]')
    ax.set_ylabel('number of users')
    ax.set_zlabel(r'optimally balanced SINR$_i$ / $\gamma_i$ [dB]', labelpad=10)
    ax.invert_yaxis()
    # 원본과 유사한 뷰 각도 설정
    ax.view_init(elev=20, azim=-120)
    ax.set_yticks(K_values)
    plt.show()


# --- 메인 실행 ---
if __name__ == "__main__":
    # 경고 메시지 단순화
    warnings.filterwarnings('ignore', category=RuntimeWarning)  # log10(0) 경고

    #plot_fig_4()
    plot_fig_5()