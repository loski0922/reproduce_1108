import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# 한글 폰트 설정 (Windows 환경 Pycharm에서 주로 사용)
# Colab이나 다른 환경에서는 폰트 경로를 알맞게 수정해야 할 수 있습니다.
def setup_korean_font():
    """
    matplotlib에서 한글을 지원하기 위한 폰트 설정을 시도합니다.
    맑은 고딕이 없는 경우, 기본 폰트를 사용합니다.
    """
    try:
        font_path = "c:/Windows/Fonts/malgun.ttf"
        font_name = fm.FontProperties(fname=font_path).get_name()
        plt.rc('font', family=font_name)
        plt.rcParams['axes.unicode_minus'] = False  # 마이너스 폰트 깨짐 방지
        print(f"'{font_name}' 폰트가 설정되었습니다.")
    except FileNotFoundError:
        print("맑은 고딕 폰트를 찾을 수 없습니다. 기본 폰트를 사용합니다.")
    except Exception as e:
        print(f"폰트 설정 중 오류 발생: {e}")


class MISO_MaxMin_SINR:
    """
    논문 "Solution of the Multiuser Downlink Beamforming..."의
    Max-min SINR balancing 알고리즘 (Table I)을 구현한 클래스.
    """

    def __init__(self, M, K, P_max, gamma_targets, noise_powers):
        """
        시뮬레이션 파라미터를 초기화합니다.

        Args:
            M (int): 기지국(BS) 안테나 수
            K (int): 사용자(User) 수
            P_max (float): 최대 총 송신 파워
            gamma_targets (np.array): 각 사용자의 타겟 SINR 비율 (K x 1)
            noise_powers (np.array): 각 사용자의 노이즈 파워 (K x 1)
        """
        self.M = M
        self.K = K
        self.P_max = P_max
        self.gamma_targets = gamma_targets
        self.noise_powers = noise_powers

        # R_all: K개의 M x M 채널 공분산 행렬 리스트
        self.R_all = []
        # R_tilde_all: 스케일링된 "가상 업링크" 채널 공분산 행렬 리스트
        self.R_tilde_all = []

        print(f"시뮬레이션 설정: M={M}, K={K}, P_max={P_max:.2f}")

    def _create_steering_vector(self, theta_deg, d_lambda=0.5):
        """
        주어진 각도(theta_deg)에 대한 ULA 조향 벡터(steering vector)를 생성합니다.
        d_lambda: 안테나 간격 (파장 대비 비율)
        """
        theta_rad = np.deg2rad(theta_deg)
        m = np.arange(self.M)
        return np.exp(-1j * 2 * np.pi * d_lambda * m * np.sin(theta_rad))

    def generate_channels(self, angular_spread_deg=0, sector_deg=120):
        """
        K명의 사용자에 대한 채널 공분산 행렬 R_i를 생성합니다.
        간단한 Rank-1 채널 (R_i = h_i * h_i^H)을 가정합니다.
        h_i는 120도 섹터 내에서 무작위 각도를 갖는 조향 벡터입니다.
        """
        print(f"{sector_deg}도 섹터 내 무작위 사용자 배치 (Rank-1 채널)")
        self.R_all = []

        # -sector_deg/2 부터 +sector_deg/2 까지 균등 분포
        user_angles = np.random.uniform(-sector_deg / 2, sector_deg / 2, self.K)

        for i in range(self.K):
            theta_i = user_angles[i]
            # 여기서는 간단히 angular_spread_deg=0 (Rank-1)만 고려
            h_i = self._create_steering_vector(theta_i)
            R_i = np.outer(h_i, h_i.conj())  # R_i = h_i * h_i^H
            self.R_all.append(R_i)

        # "가상 업링크" 채널 생성 (Duality, Section III-A)
        # R_tilde_i = R_i / sigma_i^2
        self.R_tilde_all = [
            self.R_all[i] / self.noise_powers[i] for i in range(self.K)
        ]

    def run_sinr_balancing_algorithm(self, max_iter=20, epsilon=1e-6):
        """
        논문의 Table I: SINR Balancing 알고리즘을 수행합니다.
        """
        if not self.R_tilde_all:
            print("채널이 생성되지 않았습니다. 먼저 generate_channels()를 호출하세요.")
            return

        print("Table I: SINR Balancing 알고리즘 시작...")

        # 1. 초기화 (n=0)
        n = 0
        q = np.zeros(self.K)  # 초기 업링크 파워 (논문 권장)

        # 임의의 초기 빔포머 U (정규화)
        U = np.random.randn(self.M, self.K) + 1j * np.random.randn(self.M, self.K)
        for i in range(self.K):
            U[:, i] = U[:, i] / np.linalg.norm(U[:, i])

        lambda_max = np.inf

        # Fig. 2 플롯을 위한 history 저장
        C_history = []
        min_sinr_ratio_history = []
        max_sinr_ratio_history = []

        # 4. 반복
        for n in range(1, max_iter + 1):

            # 6. 빔포머 업데이트 (U)
            # (q^(n-1)를 사용하여 U^(n) 계산)
            U_new, G_n = self._update_beamformers(q)
            U = U_new

            # 8. 파워 업데이트 (q)
            # (U^(n)을 사용하여 q^(n) 계산)
            lambda_max_new, q_new = self._update_powers(U, G_n)

            # 9. 마진(C) 계산
            C_n = 1.0 / lambda_max_new
            C_history.append(C_n)

            # (Fig. 2의 min/max SINR ratio 계산용)
            # SINR_i^{UL}(u_i^(n), q^(n-1)) / gamma_i
            # q^(n-1)은 이전 루프의 q
            # G_n은 U^(n)으로 계산된 행렬
            sinr_ratios = self._calculate_ul_sinr_ratios(G_n, q)
            min_sinr_ratio_history.append(np.min(sinr_ratios))
            max_sinr_ratio_history.append(np.max(sinr_ratios))

            # 10. 정지 조건
            if np.abs(lambda_max_new - lambda_max) < epsilon:
                print(f"반복 {n}: 수렴 완료 (Δλ < {epsilon})")
                break

            # 다음 반복을 위해 값 업데이트
            lambda_max = lambda_max_new
            q = q_new

            if n % 5 == 0 or n == 1:
                print(f"  반복 {n}: λ_max = {lambda_max:.4f}, C = {C_n:.4f}")

        else:
            print(f"최대 반복 {max_iter} 도달. 알고리즘 종료.")

        # 마지막 iteration의 SINR 값은 수렴된 q와 U를 사용해야 함
        # Fig 2는 q(n-1)과 U(n)을 사용하므로, 루프 내에서 계산한 것이 맞음
        # 단, C_history는 마지막 값 하나가 더 필요할 수 있음
        # 하지만 Fig 2는 C(n)을 플롯하므로 C_history 길이가 맞음

        # (K,) -> (n_iter,)
        # history 리스트의 길이는 n_iter
        n_iter = len(C_history)

        plot_data = {
            "iterations": np.arange(1, n_iter + 1),
            "C_margin": C_history,  # C^(n)
            "min_sinr_ratio": min_sinr_ratio_history,  # min SINR(U(n), q(n-1))
            "max_sinr_ratio": max_sinr_ratio_history  # max SINR(U(n), q(n-1))
        }

        # 최종 DL 파워 계산 (Step 11)
        p_opt = self._calculate_optimal_dl_power(U, lambda_max)
        print("알고리즘 종료.")
        print(f"최종 달성 마진 (C_opt): {C_n:.4f}")
        print(f"최종 가상 UL 파워 (q): {q}")
        print(f"최종 DL 파워 (p_opt): {p_opt}")

        return plot_data

    def _update_beamformers(self, q_prev):
        """
        (Table I - Step 6, 7)
        고정된 파워(q_prev)에 대해 빔포머(U)를 업데이트합니다.
        각 사용자 i에 대해 일반화 고유값 문제를 풉니다.
        """
        U_new = np.zeros((self.M, self.K), dtype=complex)

        for i in range(self.K):
            # 1. 간섭+노이즈 공분산 Q_i 계산 (Eq 31)
            # (가상 노이즈 = 1, 따라서 I 추가)
            Q_i = np.eye(self.M, dtype=complex)
            for k in range(self.K):
                if i != k:
                    Q_i += q_prev[k] * self.R_tilde_all[k]

            # 2. 일반화 고유값 문제 풀이 (Eq 30)
            # R_tilde_i * u_i = lambda * Q_i * u_i
            # scipy.linalg.eigh는 (A, B)에 대해 A*x = lambda*B*x 를 풉니다.
            try:
                eigvals, eigvecs = scipy.linalg.eigh(self.R_tilde_all[i], Q_i)

                # 3. 최대 고유값에 해당하는 고유벡터(빔포머) 선택
                u_i = eigvecs[:, -1]  # 최대 고유값은 마지막에 위치

                # 4. 정규화 (Step 7)
                U_new[:, i] = u_i / np.linalg.norm(u_i)

            except scipy.linalg.LinAlgError:
                print(f"경고: 사용자 {i}의 고유값 계산 실패. 이전 빔포머 유지.")
                # 에러 발생 시 (Q_i가 singular 등) 이전 U 값을 유지 (실제로는 U_prev 필요)
                # 여기서는 간단히 랜덤 벡터 또는 영벡터가 될 수 있으므로 주의
                # 간단한 구현을 위해 정규화된 랜덤 벡터 사용
                u_i = np.random.randn(self.M) + 1j * np.random.randn(self.M)
                U_new[:, i] = u_i / np.linalg.norm(u_i)

        # G 행렬 계산 (파워 업데이트 단계에서 필요)
        # G_ik = u_i^H * R_tilde_k * u_i
        G = np.zeros((self.K, self.K), dtype=complex)
        for i in range(self.K):
            u_i = U_new[:, i]
            for k in range(self.K):
                G[i, k] = u_i.conj().T @ self.R_tilde_all[k] @ u_i

        return U_new, np.abs(G)  # G는 실수(파워)이므로 절대값 사용

    def _update_powers(self, U, G):
        """
        (Table I - Step 8)
        고정된 빔포머(U)에 대해 파워(q)를 업데이트합니다.
        확장 결합 행렬 Lambda의 지배 고유벡터(dominant eigenvector)를 찾습니다.
        """

        # 1. 행렬 D 계산
        # G_ii = u_i^H * R_tilde_i * u_i
        G_ii = np.diag(G)
        # 0으로 나누는 것을 방지
        G_ii[G_ii == 0] = 1e-9
        D = np.diag(self.gamma_targets / G_ii)

        # 2. 행렬 Psi^T (UL crosstalk) 계산
        # Psi_T[i, k] = G_ik (i != k)
        Psi_T = G.copy()
        np.fill_diagonal(Psi_T, 0)

        # 3. 확장 결합 행렬 Lambda (Eq 16) 생성

        # A = D * Psi^T
        A = D @ Psi_T

        # b = D * sigma (가상 노이즈 sigma = [1, ..., 1]^T)
        b = D @ np.ones(self.K)

        Lambda = np.zeros((self.K + 1, self.K + 1))

        # Top-left (K x K)
        Lambda[0:self.K, 0:self.K] = A
        # Top-right (K x 1)
        Lambda[0:self.K, self.K] = b
        # Bottom-left (1 x K)
        Lambda[self.K, 0:self.K] = (1.0 / self.P_max) * (np.ones(self.K) @ A)
        # Bottom-right (1 x 1)
        Lambda[self.K, self.K] = (1.0 / self.P_max) * (np.ones(self.K) @ b)

        # 4. Lambda의 지배 고유값(lambda_max) 및 고유벡터(q_ext) 계산
        # Lambda는 비대칭일 수 있으므로 np.linalg.eig 사용
        try:
            eigvals, eigvecs = np.linalg.eig(Lambda)

            # Perron-Frobenius 정리에 따라, lambda_max는 실수이고 양수
            lambda_max_idx = np.argmax(np.real(eigvals))
            lambda_max = np.real(eigvals[lambda_max_idx])

            # 고유벡터는 양수여야 함
            q_ext = np.real(eigvecs[:, lambda_max_idx])

            # 5. q_ext 스케일링 (마지막 요소가 1이 되도록)
            if q_ext[-1] != 0:
                q_ext = q_ext / q_ext[-1]
            else:
                # 마지막 요소가 0인 경우 (비정상), 강제로 1로 설정 (오류 처리)
                q_ext[-1] = 1.0
                print("경고: q_ext의 마지막 요소가 0입니다.")

            # 6. q 갱신
            q = q_ext[0:self.K]
            q[q < 0] = 0  # 파워는 음수가 될 수 없음

            return lambda_max, q

        except np.linalg.LinAlgError:
            print("경고: Lambda의 고유값 계산 실패. 이전 파워 유지.")
            return np.inf, np.zeros(self.K)  # 오류 시

    def _calculate_ul_sinr_ratios(self, G, q):
        """
        Fig. 2의 min/max 바 계산용.
        SINR_i^{UL}(u_i^(n), q^(n-1)) / gamma_i 를 계산합니다.

        Args:
            G (np.array): G_ik = |u_i^(n)H * R_tilde_k * u_i^(n)| (K x K)
            q (np.array): q^(n-1) (K,)
        """
        sinr_ratios = np.zeros(self.K)

        for i in range(self.K):
            G_ii = G[i, i]
            interference = 0.0
            for k in range(self.K):
                if i != k:
                    interference += q[k] * G[i, k]

            # 가상 노이즈 = 1.0
            sinr_ul_i = (q[i] * G_ii) / (interference + 1.0)

            if self.gamma_targets[i] > 0:
                sinr_ratios[i] = sinr_ul_i / self.gamma_targets[i]
            else:
                sinr_ratios[i] = np.inf if sinr_ul_i > 0 else 0

        # q가 0일 수 있으므로 SINR이 0/0 -> nan이 될 수 있음
        sinr_ratios = np.nan_to_num(sinr_ratios, nan=0.0)
        return sinr_ratios

    def _calculate_optimal_dl_power(self, U, lambda_max_converged):
        """
        (Table I - Step 11)
        수렴된 U와 lambda_max를 사용하여 최적의 다운링크 파워 p_opt를 계산합니다.
        이는 Lambda 대신Upsilon 행렬을 사용하여 동일한 고유값 문제를 푸는 것과 같습니다.
        """
        # Upsilon 계산을 위해 D와 Psi가 필요

        # G_ik = u_i^H * R_tilde_k * u_i
        G = np.zeros((self.K, self.K), dtype=complex)
        for i in range(self.K):
            u_i = U[:, i]
            for k in range(self.K):
                G[i, k] = u_i.conj().T @ self.R_tilde_all[k] @ u_i
        G = np.abs(G)

        G_ii = np.diag(G)
        G_ii[G_ii == 0] = 1e-9
        D = np.diag(self.gamma_targets / G_ii)

        # Psi (DL crosstalk) 계산, Psi[k, i] = G_ki (k != i)
        Psi = G.T.copy()  # Psi는 Psi^T의 transpose
        np.fill_diagonal(Psi, 0)

        # A_dl = D * Psi
        A_dl = D @ Psi

        # b_dl = D * sigma
        b_dl = D @ np.ones(self.K)

        # Upsilon (Eq 12)
        Upsilon = np.zeros((self.K + 1, self.K + 1))
        Upsilon[0:self.K, 0:self.K] = A_dl
        Upsilon[0:self.K, self.K] = b_dl
        Upsilon[self.K, 0:self.K] = (1.0 / self.P_max) * (np.ones(self.K) @ A_dl)
        Upsilon[self.K, self.K] = (1.0 / self.P_max) * (np.ones(self.K) @ b_dl)

        # 고유값 문제는 풀 필요 없음 (lambda_max는 동일)
        # p_ext는 (Upsilon - lambda_max * I) * p_ext = 0 의 해
        # (A_dl - lambda_max * I) * p = -b_dl (마지막 행 무시하고 p_ext = [p, 1]^T 가정 시)
        # p = (lambda_max * I - A_dl)^-1 * b_dl

        try:
            I_K = np.eye(self.K)
            p_opt = np.linalg.solve(lambda_max_converged * I_K - A_dl, b_dl)
            p_opt[p_opt < 0] = 0  # 파워는 양수
            return p_opt
        except np.linalg.LinAlgError:
            print("경고: DL 파워 계산 실패 (Singular matrix)")
            return np.zeros(self.K)

    def plot_convergence(self, plot_data):
        """
        Fig. 2와 유사한 수렴 그래프를 그립니다.
        """
        setup_korean_font()  # 한글 폰트 설정

        iters = plot_data["iterations"]
        C_margin = plot_data["C_margin"]
        min_sinr = plot_data["min_sinr_ratio"]
        max_sinr = plot_data["max_sinr_ratio"]

        n_iters = len(iters)
        if n_iters == 0:
            print("플롯할 데이터가 없습니다.")
            return

        plt.figure(figsize=(10, 6))

        # 로그 스케일이 더 적절할 수 있으나, Fig. 2는 선형 스케일
        # plt.yscale('log')

        width = 0.25

        plt.bar(iters - width, min_sinr, width,
                label=r'$min_i \frac{SINR_i^{UL}(\mathbf{u}_i^{(n)}, \mathbf{q}^{(n-1)})}{\gamma_i}$',
                color='C0', hatch='//', edgecolor='black')

        plt.bar(iters, C_margin, width,
                label=r'$C^{(n)} = C^{DL}(\mathbf{U}^{(n)}, P_{max})$',
                color='C1', edgecolor='black')

        plt.bar(iters + width, max_sinr, width,
                label=r'$max_i \frac{SINR_i^{UL}(\mathbf{u}_i^{(n)}, \mathbf{q}^{(n-1)})}{\gamma_i}$',
                color='C2', edgecolor='black')

        # 수렴선을 C_margin의 마지막 값으로 그림
        plt.axhline(y=C_margin[-1], color='r', linestyle='--', label=f'수렴 값 (C={C_margin[-1]:.3f})')

        plt.xlabel("반복 (Iteration n)", fontsize=12)
        plt.ylabel("SINR / Target (비율)", fontsize=12)
        plt.title(f"알고리즘 수렴 과정 (Fig. 2 재현) - M={self.M}, K={self.K}", fontsize=14)
        plt.xticks(iters[::max(1, n_iters // 10)])  # x축 레이블 적절히 조절
        plt.legend()
        plt.grid(axis='y', linestyle=':', alpha=0.7)
        plt.tight_layout()

        # 그래프를 'convergence_plot.png' 파일로 저장
        plot_filename = "convergence_plot.png"
        plt.savefig(plot_filename)
        print(f"수렴 그래프가 '{plot_filename}' 파일로 저장되었습니다.")
        plt.show()


# --- 메인 실행 블록 ---
if __name__ == "__main__":

    # --- 시뮬레이션 파라미터 설정 ---
    # (논문의 Fig 2/4/5/6의 정확한 값은 명시되지 않음)

    M_ANTENNAS = 4  # 기지국 안테나 수 (M)
    K_USERS = 3  # 사용자 수 (K)

    # 총 파워 (dBm -> linear)
    P_MAX_DBM = 10.0
    P_MAX_LINEAR = 10 ** (P_MAX_DBM / 10.0)

    # 모든 사용자가 동일한 타겟 SINR 비율을 갖는다고 가정
    # (P1 문제에서는 gamma 자체보다 비율이 중요)
    TARGET_GAMMAS = np.ones(K_USERS)

    # 노이즈 파워 (dBm -> linear)
    # (SNR = P_max / noise, 여기서는 임의의 값 설정)
    NOISE_POWER_DBM = -10.0
    NOISE_POWER_LINEAR = 10 ** (NOISE_POWER_DBM / 10.0)

    # 모든 사용자가 동일한 노이즈 파워를 갖는다고 가정
    # (논문은 불균등 노이즈도 가상 업링크로 처리 가능함을 보임)
    NOISE_POWERS = np.ones(K_USERS) * NOISE_POWER_LINEAR

    # --- 시뮬레이션 실행 ---

    # 1. 시뮬레이터 인스턴스 생성
    beamformer = MISO_MaxMin_SINR(
        M=M_ANTENNAS,
        K=K_USERS,
        P_max=P_MAX_LINEAR,
        gamma_targets=TARGET_GAMMAS,
        noise_powers=NOISE_POWERS
    )

    # 2. 채널 생성 (무작위)
    np.random.seed(42)  # 재현을 위한 시드 고정
    beamformer.generate_channels()

    # 3. 알고리즘 실행
    plot_data = beamformer.run_sinr_balancing_algorithm(max_iter=10)

    # 4. 결과 플로팅
    if plot_data:
        beamformer.plot_convergence(plot_data)