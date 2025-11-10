function [U_opt, p_opt, P_min, C_final, feasible] = solve_power_minimization(R_cell, gamma_vec, P_max, sigma_sq_vec, epsilon, max_iter)
    % 이 함수는 "Solution of the Multiuser Downlink Beamforming Problem..." (Schubert and Boche, 2004)
    % 논문의 Table II 에 제시된 전력 최소화 알고리즘을 구현합니다.
    %
    % 입력:
    %   R_cell       - Kx1 cell array, 각 셀은 사용자 i의 M x M 채널 공분산 행렬 (R_i)
    %   gamma_vec    - Kx1 vector, 각 사용자의 타겟 SINR (gamma_i) 
    %   P_max        - 스칼라, 최대 총 송신 전력 (P_max) 
    %   sigma_sq_vec - Kx1 vector, 각 사용자의 수신기 잡음 전력 (sigma_i^2) 
    %   epsilon      - 스칼라, 수렴 조건 (e.g., 1e-5) 
    %   max_iter     - 스칼라, 최대 반복 횟수
    %
    % 출력:
    %   U_opt        - M x K matrix, 최적 빔포밍 벡터 
    %   p_opt        - Kx1 vector, 최적 다운링크 전력 할당 (Line 16)
    %   P_min        - 스칼라, 최소화된 총 전력 (2단계에서 계산됨) (Line 13)
    %   C_final      - 스칼라, 최종 SINR 마진 (1단계에서 계산됨) (Line 9)
    %   feasible     - 논리값, SINR 목표 달성 가능 여부

    % --- 초기화 (Table II, Lines 1-3) ---
    K = length(gamma_vec);      % 사용자 수 (K)
    M = size(R_cell{1}, 1);     % 안테나 수 (M)
    
    n = 0;
    q = zeros(K, 1);            % 가상 업링크 전력 벡터 (Line 1)
    q_prev = q;
    C = 0;                      % SINR 마진 (Line 1)
    U = zeros(M, K);
    
    % 정규화된 공분산 행렬 (R_tilde_i = R_i / sigma_i^2) (Line 2)
    R_tilde_cell = cell(K, 1);
    for i = 1:K
        R_tilde_cell{i} = R_cell{i} / sigma_sq_vec(i);
    end
    % 잡음은 1로 정규화됨 (sigma_i^2 = 1) (Line 3)
    noise_vec = ones(K, 1); 

    disp('알고리즘 반복 시작...');

    % --- 반복 (Table II, Line 4) ---
    while n < max_iter
        n = n + 1;
        
        % --- 빔포밍 벡터 업데이트 (Table II, Lines 5-6) ---
        % 각 사용자에 대해 일반화 고유값 문제 풀이
        for i = 1:K
            % 간섭 공분산 행렬 Q_i 계산 
            Q_i = zeros(M, M);
            for k = 1:K
                if i ~= k
                    Q_i = Q_i + q(k) * R_tilde_cell{k};
                end
            end
            Q_i = Q_i + eye(M); % 정규화된 잡음 추가 (I) 
            
            % u_i = v_max(R_tilde_i, Q_i) (Line 5)
            % eigs를 사용하여 가장 큰 고유값에 해당하는 고유벡터 찾기
            [u_vec, ~] = eigs(R_tilde_cell{i}, Q_i, 1, 'largestabs');
            
            U(:, i) = u_vec / norm(u_vec); % 정규화 (Line 6)
        end
        
        % --- D 행렬 및 Psi^T (전치된 커플링 행렬) 계산 ---
        % D = diag{gamma_i / (u_i^H * R_tilde_i * u_i)} 
        diag_D = zeros(K, 1);
        for i = 1:K
            diag_D(i) = gamma_vec(i) / (U(:,i)' * R_tilde_cell{i} * U(:,i));
        end
        D = diag(diag_D);
        
        % Psi^T (업링크 커플링 행렬) 
        % Psi(i, k) = |u_i^H * R_k * u_i|^2 / (u_i^H * R_i * u_i) (정규화 전)
        % 여기서는 정규화된 R_tilde 사용
        Psi_T = zeros(K, K);
        for i = 1:K
            for k = 1:K
                if i ~= k
                    % Psi_T(i, k)는 사용자 k -> 사용자 i 간섭 (업링크 관점)
                    Psi_T(i, k) = (abs(U(:,i)' * R_tilde_cell{k} * U(:,i))) * diag_D(i);
                end
            end
        end

        % --- 전력 제어 (Table II, Lines 7-14) ---
        if C < 1 % 1단계: SINR 마진 최대화 (Feasibility test) (Line 7)
            
            % 확장된 커플링 행렬 Lambda(U, P_max) 구축 
            Lambda_11 = D * Psi_T; % 논문 표기법 D * Psi^T(U) 
            Lambda_12 = D * noise_vec; % D * sigma 
            Lambda_21 = (1/P_max) * ones(1, K) * Lambda_11; % (1/P_max) * 1^T * D * Psi^T 
            Lambda_22 = (1/P_max) * ones(1, K) * Lambda_12; % (1/P_max) * 1^T * D * sigma 
            
            Lambda = [Lambda_11, Lambda_12; Lambda_21, Lambda_22];
            
            % 최대 고유값 및 고유벡터 계산 (Line 8)
            try
                [q_ext, lambda_max] = eigs(Lambda, 1, 'largestabs');
                lambda_max = real(lambda_max); % 실수부만 사용
                
                C = 1 / lambda_max; % (Line 9)
                
                % q 벡터 업데이트 (마지막 요소로 정규화) 
                q = real(q_ext(1:K) / q_ext(K+1));
                q(q < 0) = 0; % 물리적으로 음수 전력은 0으로 처리

            catch ME
                disp('Lambda 행렬 고유값 계산 실패. 비현실적인 채널일 수 있습니다.');
                feasible = false;
                U_opt = U;
                p_opt = zeros(K, 1);
                P_min = inf;
                C_final = C;
                return;
            end
            
        else % 2단계: 총 전력 최소화 (Line 11)
            
            % (I - D*Psi^T)^-1 * D * 1 (Line 12)
            try
                A = eye(K) - D * Psi_T;
                b = D * noise_vec;
                q = A \ b; % 선형 시스템 풀이
                q(q < 0) = 0;

            catch ME
                disp('전력 최소화 (2단계) 계산 실패. 행렬이 특이(singular)할 수 있습니다.');
                % 1단계에서 C >= 1 이었으므로 feasible은 맞음
                feasible = true; 
                U_opt = U; 
                p_opt = zeros(K,1); % 2단계 p_opt 계산 전 실패
                P_min = sum(q_prev); % 이전 단계의 전력
                C_final = C;
                return;
            end
        end
        
        % --- 수렴 조건 확인 (Table II, Line 15) ---
        % 논문에서는 SINR이 균형을 이루는 것을 기준으로 함 
        % 여기서는 q 벡터의 변화량으로 대체
        if norm(q - q_prev) < epsilon
            disp(['수렴 완료 (Iteration: ' num2str(n) ')']);
            break;
        end
        q_prev = q;
        
        if n == max_iter
            disp('최대 반복 횟수에 도달했습니다.');
        end
    end

    % --- 최종 결과 할당 ---
    U_opt = U;
    C_final = C;

    if C_final < 1
        % 1단계에서 수렴했으나 C < 1 이면 달성 불가능
        feasible = false;
        p_opt = zeros(K, 1);
        P_min = inf;
        disp('SINR 목표 달성 불가능 (Infeasible).');
    else
        % 2단계에서 수렴했거나, 1단계에서 C >= 1 로 수렴
        feasible = true;
        P_min = sum(q); % 2단계에서 계산된 q의 합 (Line 13)
        
        % --- 최종 다운링크 전력 계산 (Table II, Line 16) ---
        % p_opt = (I - D*Psi)^-1 * D * 1 (Line 16)
        % Psi (다운링크 커플링 행렬) 
        Psi = zeros(K, K);
        for i = 1:K
            for k = 1:K
                if i ~= k
                    % Psi(i, k)는 사용자 k -> 사용자 i 간섭 (다운링크)
                    Psi(i, k) = (abs(U(:,i)' * R_tilde_cell{k} * U(:,k)))^2 * diag_D(i);
                end
            end
        end
        
        try
            A_dl = eye(K) - D * Psi;
            b_dl = D * noise_vec;
            p_opt = A_dl \ b_dl; % p_opt 계산 (Line 16)
            p_opt(p_opt < 0) = 0;
            
            % 계산된 p_opt의 실제 총합 (P_min과 유사해야 함)
            P_dl_total = sum(p_opt);
            disp(['최소 필요 전력 (업링크 기준): ' num2str(P_min)]);
            disp(['최소 필요 전력 (다운링크 기준): ' num2str(P_dl_total)]);
            P_min = P_dl_total; % 다운링크 기준으로 최종 업데이트

        catch ME
             disp('최종 다운링크 전력(p_opt) 계산 실패.');
             p_opt = zeros(K, 1);
        end
    end
end