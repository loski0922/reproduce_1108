% --- 시뮬레이션 파라미터 ---
M = 4; % 기지국 안테나 수 
K = 3; % 사용자 수 (K=5 대신 3으로 단순화)
P_max_dBm = 20; % 최대 전력 (dBm) 
noise_dBm = -90; % 잡음 전력 (dBm)

% 선형 스케일로 변환
P_max_W = 10^(P_max_dBm / 10) / 1000;
noise_W = 10^(noise_dBm / 10) / 1000;
sigma_sq_vec = ones(K, 1) * noise_W;

% 타겟 SINR (dB)
target_sinr_dB = 5; % 
gamma_vec = ones(K, 1) * 10^(target_sinr_dB / 10);

% --- 임의의 채널 공분산 행렬 생성 (테스트용) ---
% 실제로는 채널 추정을 통해 얻어져야 함 
R_cell = cell(K, 1);
for i = 1:K
    % 임의의 M x M 에르미트 행렬 생성
    H = (randn(M, M) + 1j * randn(M, M)) / sqrt(2);
    R_cell{i} = H * H'; % R_i = E{h_i * h_i^H} 
end

% --- 알고리즘 실행 ---
epsilon = 1e-6;
max_iter = 100;

[U_opt, p_opt, P_min, C_final, feasible] = ...
    solve_power_minimization(R_cell, gamma_vec, P_max_W, sigma_sq_vec, epsilon, max_iter);

% --- 결과 출력 ---
if feasible
    disp('결과: SINR 목표 달성 가능');
    disp(['최소 필요 총 전력: ' num2str(10 * log10(P_min * 1000)) ' dBm']);
    disp('최적 빔포밍 행렬 U_opt (M x K):');
    disp(U_opt);
    disp('최적 다운링크 전력 p_opt (mW):');
    disp(p_opt * 1000);
else
    disp('결과: SINR 목표 달성 불가능 (Infeasible)');
    disp(['현재 P_max로 달성 가능한 최대 SINR 마진 (C): ' num2str(C_final)]);
end