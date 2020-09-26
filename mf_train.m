function [U, V] = mf_train(R, U, V, U_reg, V_reg, epsilon, max_iter, R_test, solver, env)
    % function [U, V] = mf_train(R, U, V, U_reg, V_reg, epsilon, max_iter, R_test)
    % Inputs:
    %   R: the sparse (n-by-m) rating matrix
    %   U, V: the interaction (d-by-m) and (d-by-n) matrices, where d is latent vector size.
    %   U_reg, V_reg: the frequncy-aware regularization coefficients of the two interaction matrices.
    %   epsilon: stopping tolerance in (0,1). Use a larger value if the training time is too long.
    %   max_iter: the maximal number of newton iteration will be run.
    %   R_test: the sparse (n-by-m) testing rating matrix
    %   solve: use 'gauss' for Gauss-Newton method and 'alscg' for alternating least square with conjugate gradient method.
    %   env: indicate use 'cpu' or 'gpu' solver
    % Outputs:
    %   U, V: the interaction (d-by-m) and (d-by-n) matrices.

    print(solver, env);

    global get_embedding_inner get_cross_embedding_inner;

    if strcmp(env, 'gpu')
        U = gpuArray(U);
        V = gpuArray(V);
        U_reg = gpuArray(U_reg);
        V_reg = gpuArray(V_reg);
        get_embedding_inner = @get_embedding_inner_gpu;
        get_cross_embedding_inner = @get_cross_embedding_inner_gpu;
    else
        get_embedding_inner = @get_embedding_inner_cpu;
        get_cross_embedding_inner = @get_cross_embedding_inner_cpu;
    end

    if strcmp(solver, 'gauss')
        [U V] = gauss_newton_solver(R, U, V, U_reg, V_reg, epsilon, max_iter, R_test);
    elseif strcmp(solver, 'alscg')
        [U V] = alternating_least_square_cg_solver(R, U, V, U_reg, V_reg, epsilon, max_iter, R_test);
    end

end

function [U V] = alternating_least_square_cg_solver(R, U, V, U_reg, V_reg, epsilon, max_iter, R_test, solver)
    global get_embedding_inner;
    [i_idx_R, j_idx_R, vals_R] = find(R);
    [i_idx_R_test, j_idx_R_test, vals_R_test] = find(R_test);
    R_idx = [i_idx_R, j_idx_R];
    R_test_idx = [i_idx_R_test, j_idx_R_test];

    total_t = 0;

    for k = 1:max_iter

        if (k == 1)
            B = get_embedding_inner(U, V, i_idx_R, j_idx_R) - R;
        end

        G = [U .* U_reg' V .* V_reg'] + [V * B' U * B];

        if (k == 1)
            G_norm_0 = norm(G, 'fro');
        elseif (norm(G, 'fro') <= epsilon * G_norm_0)
            fprintf('Newton stopping condition');
            break;
        end

        tic;
        [U, B, cg_iters_U] = update_block_alscg(U, V, B, U_reg, R_idx, 'no_transposed');
        [V, B, cg_iters_V] = update_block_alscg(V, U, B, V_reg, R_idx, 'transposed');
        total_t = total_t + toc;

        print_results_alscg(k, total_t, cg_iters_U, cg_iters_V, U, V, R_test, R_test_idx, U_reg, V_reg, B);
    end

    if (k == max_iter)
        fprintf('Warning: reach max training iteration. Terminate training process.\n');
    end

end

function [U, B, cg_iters] = update_block_alscg(U, V, B, reg, R_idx, option)
    global get_embedding_inner;
    eta = 0.3;
    cg_max_iter = 20;
    Su = zeros(size(U));

    if strcmp(option, 'transposed')
        C =- (U .* reg' + V * B);
    else
        C =- (U .* reg' + V * B');
    end

    D = C;
    gamma_0 = sum(sum(C .* C));
    gamma = gamma_0;
    cg_iters = 0;

    while (gamma > eta * eta * gamma_0)
        cg_iters = cg_iters + 1;

        if strcmp(option, 'transposed')
            Z = get_embedding_inner(V, D, R_idx(:, 1), R_idx(:, 2));
            Dh = D .* reg' + V * Z;
        else
            Z = get_embedding_inner(D, V, R_idx(:, 1), R_idx(:, 2));
            Dh = D .* reg' + V * Z';
        end

        alpha = gamma / sum(sum(D .* Dh));
        Su = Su + alpha * D;
        C = C - alpha * Dh;
        gamma_new = sum(sum(C .* C));
        beta = gamma_new / gamma;
        D = C + beta * D;
        gamma = gamma_new;

        if (cg_iters >= cg_max_iter)
            fprintf('Warning: reach max CG iteration. CG process is terminated.\n');
            break;
        end

    end

    if strcmp(option, 'transposed')
        Delta = get_embedding_inner(V, Su, R_idx(:, 1), R_idx(:, 2));
    else
        Delta = get_embedding_inner(Su, V, R_idx(:, 1), R_idx(:, 2));
    end

    U = U + Su;
    B = B + Delta;
end

function [U, V] = gauss_newton_solver(R, U, V, U_reg, V_reg, epsilon, max_iter, R_test, solver)
    global get_embedding_inner get_cross_embedding_inner;
    [i_idx_R, j_idx_R, vals_R] = find(R);
    [i_idx_R_test, j_idx_R_test, vals_R_test] = find(R_test);
    R_idx = [i_idx_R, j_idx_R];
    R_test_idx = [i_idx_R_test, j_idx_R_test];
    total_t = 0;

    for k = 1:max_iter

        if (k == 1)
            B = get_embedding_inner(U, V, i_idx_R, j_idx_R) - R;
        end

        G = [U .* U_reg' V .* V_reg'] + [V * B' U * B];

        if (k == 1)
            G_norm_0 = norm(G, 'fro');
        elseif (norm(G, 'fro') <= epsilon * G_norm_0)
            fprintf('Newton stopping condition');
            break;
        end

        tic;
        nu = 0.1;
        min_step_size = 1e-20;
        [Su, Sv, cg_iters] = cg(U, V, G, U_reg, V_reg, R_idx);
        Delta_1 = get_cross_embedding_inner(Su, Sv, U, V, R_idx(:, 1), R_idx(:, 2));
        Delta_2 = get_embedding_inner(Su, Sv, R_idx(:, 1), R_idx(:, 2));
        US_u = sum(U .* Su) * U_reg;
        VS_v = sum(V .* Sv) * V_reg;
        SS = sum([Su Sv] .* [Su Sv]) * [U_reg; V_reg];
        GS = sum(sum(G .* [Su Sv]));
        theta = 1;
        B_norm = norm(B, 'fro')^2;

        for ls_steps = 0:intmax;

            if (theta < min_step_size)
                fprintf('Warning: step size is too small in line search. Switch to the next block of variables.\n');
                return;
            end

            B_new = B + theta * Delta_1 + theta * theta * Delta_2;
            B_new_norm = norm(B_new, 'fro')^2;
            f_diff = 0.5 * (2 * theta * (US_u + VS_v) + theta * theta * SS) + 0.5 * (B_new_norm - B_norm);

            if (f_diff <= nu * theta * GS)
                U = U + theta * Su;
                V = V + theta * Sv;
                B = B_new;
                break;
            end

            theta = theta * 0.5;
        end

        total_t = total_t + toc;

        print_results_gauss(k, total_t, cg_iters, ls_steps, U, V, R_test, R_test_idx, U_reg, V_reg, B);
    end

    if (k == max_iter)
        fprintf('Warning: reach max training iteration. Terminate training process.\n');
    end

end

function [Su, Sv, cg_iters] = cg(U, V, G, U_reg, V_reg, R_idx)
    global get_embedding_inner get_cross_embedding_inner;
    m = size(U, 2);
    n = size(V, 2);
    eta = 0.3;
    cg_max_iter = 20;
    S = zeros(size(G));
    C = -G;
    D = C;
    gamma_0 = sum(sum(C .* C));
    gamma = gamma_0;
    cg_iters = 0;

    while (gamma > eta * eta * gamma_0)
        cg_iters = cg_iters + 1;
        Z = get_cross_embedding_inner(D(:, 1:m), D(:, m + 1:end), U, V, R_idx(:, 1), R_idx(:, 2));
        Dh = D .* [U_reg; V_reg]' + [V * Z' U * Z];
        alpha = gamma / sum(sum(D .* Dh));
        S = S + alpha * D;
        C = C - alpha * Dh;
        gamma_new = sum(sum(C .* C));
        beta = gamma_new / gamma;
        D = C + beta * D;
        gamma = gamma_new;

        if (cg_iters >= cg_max_iter)
            fprintf('Warning: reach max CG iteration. CG process is terminated.\n');
            break;
        end

    end

    Su = S(:, 1:m);
    Sv = S(:, m + 1:end);
end

%point wise summation
%z_(m,n) = v_n^T*s_u^m + u_m^T*s_v^n
function Z = get_cross_embedding_inner_gpu(Su, Sv, U, V, i_idx, j_idx)
    l = size(i_idx, 1);
    vals = gpuArray(zeros(1, l));

    num_batches = 10;
    bsize = ceil(l / num_batches);

    for i = 1:num_batches
        range = (i - 1) * bsize + 1:min(l, i * bsize);
        vals(range) = dot(V(:, j_idx(range)), Su(:, i_idx(range))) + dot(Sv(:, j_idx(range)), U(:, i_idx(range)));
    end

    Z = sparse(i_idx, j_idx, vals, size(U, 2), size(V, 2));
end

%point wise summation
% z_(m,n) = u_m^T*v_n
function Z = get_embedding_inner_gpu(U, V, i_idx, j_idx)
    l = size(i_idx, 1);
    vals = gpuArray(zeros(1, l));

    num_batches = 10;
    bsize = ceil(l / num_batches);

    for i = 1:num_batches
        range = (i - 1) * bsize + 1:min(l, i * bsize);
        vals(range) = dot(V(:, j_idx(range)), U(:, i_idx(range)));
    end

    Z = sparse(i_idx, j_idx, vals, size(U, 2), size(V, 2));
end

function [] = print(solver, env)
    fprintf('Using ''%5s'' solver with ''%3s'' env.\n', solver, env);

    if strcmp(solver, 'gauss')
        fprintf('%4s  %15s  %3s  %3s  %15s  %15s  %15s  %15s  %15s\n', 'iter', 'time', '#cg', '#ls', 'obj', '|G|', 'test_loss', '|G_U|', '|G_V|', 'loss');
    elseif strcmp(solver, 'alscg')
        fprintf('%4s  %15s  %3s  %3s  %15s  %15s  %15s  %15s  %15s  %15s\n', 'iter', 'time', '#cg_U', '#cg_V', 'obj', '|G_U|', '|G_V|', 'test_loss', '|G|', 'loss');
    elseif strcmp(solver, 'als')
        fprintf('%4s  %15s  %15s  %15s  %15s\n', 'iter', 'time', 'obj', 'test_loss', 'loss');
    end

end

function [] = print_results_gauss(k, total_t, cg_iters, ls_steps, U, V, R_test, R_test_idx, U_reg, V_reg, B)
    global get_embedding_inner;
    m = size(U, 2);
    n = size(V, 2);

    % Function value
    loss = 0.5 * full(sum(sum(B .* B)));
    f = 0.5 * (sum(U .* U) * U_reg + sum(V .* V) * V_reg) + loss;

    % Test Loss
    Y_test_tilde = get_embedding_inner(U, V, R_test_idx(:, 1), R_test_idx(:, 2));
    test_loss = sqrt(full(sum(sum((R_test - Y_test_tilde) .* (R_test - Y_test_tilde)))) / size(R_test_idx, 1));

    % Gradient
    G = [U .* U_reg' V .* V_reg'] + [V * B' U * B];
    G_norm = norm(G, 'fro');
    GU_norm = norm(G(:, 1:m), 'fro');
    GV_norm = norm(G(:, m + 1:end), 'fro');

    fprintf('%4d  %15.3f  %3d  %3d  %15.3f  %15.6f  %15.6f  %15.6f  %15.6f  %15.3f\n', k, total_t, cg_iters, ls_steps, f, G_norm, test_loss, GU_norm, GV_norm, loss);
end

function [] = print_results_alscg(k, total_t, cg_iters_U, cg_iters_V, U, V, R_test, R_test_idx, U_reg, V_reg, B)
    global get_embedding_inner;
    m = size(U, 2);
    n = size(V, 2);

    % Function value
    loss = 0.5 * full(sum(sum(B .* B)));
    f = 0.5 * (sum(U .* U) * U_reg + sum(V .* V) * V_reg) + loss;

    % Test Loss
    Y_test_tilde = get_embedding_inner(U, V, R_test_idx(:, 1), R_test_idx(:, 2));
    test_loss = sqrt(full(sum(sum((R_test - Y_test_tilde) .* (R_test - Y_test_tilde)))) / size(R_test_idx, 1));

    % Gradient
    G = [U .* U_reg' V .* V_reg'] + [V * B' U * B];
    G_norm = norm(G, 'fro');
    GU_norm = norm(G(:, 1:m), 'fro');
    GV_norm = norm(G(:, m + 1:end), 'fro');
    GU = U .* U_reg' + V * B';
    GV = V .* V_reg' + U * B;

    fprintf('%4d  %15.3f  %3d  %3d  %15.3f  %15.6f  %15.6f  %15.6f  %15.6f  %15.3f\n', k, total_t, cg_iters_U, cg_iters_V, f, GU_norm, GV_norm, test_loss, G_norm, loss);
end
