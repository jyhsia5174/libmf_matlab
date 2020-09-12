function [U, V] = mm_train(R, U, V, U_reg, V_reg, epsilon, max_iter, R_test, d, solver, env)
    % function [U, V] = fm_train(R, U, V, U_reg, V_reg, epsilon, max_iter, R_test)
    % Inputs:
    %   R: rating matrix
    %   U_reg, V_reg: the frequncy-aware regularization coefficients of the two interaction matrices.
    %   epsilon: stopping tolerance in (0,1). Use a larger value if the training time is too long.
    %   R_test: testing rating matrix
    %   U, V: the interaction (d-by-n) matrices.
    % Outputs:
    %   U, V: the interaction (d-by-n) matrices.

    [m, n] = size(R);
    l = nnz(R);
    nnz_R_test = nnz(R_test);

    [i_idx_R, j_idx_R, vals_R] = find(R);
    [i_idx_R_test, j_idx_R_test, vals_R_test] = find(R_test);
    total_t = 0;

    if strcmp(solver, 'als')
        uni_i_idx_R = unique(i_idx_R);
        uni_j_idx_R = unique(j_idx_R);
        U = U(1:d, :);
        V = V(1:d, :);

        for i = 1:length(uni_i_idx_R)
            m2ns{uni_i_idx_R(i)} = find(R(uni_i_idx_R(i), :));
        end

        for i = 1:length(uni_j_idx_R)
            n2ms{uni_j_idx_R(i)} = find(R(:, uni_j_idx_R(i))');
        end

    end

    if strcmp(env, 'gpu')
        R = gpuArray(R);
        R_test = gpuArray(R_test);
        U = gpuArray(U);
        V = gpuArray(V);
        U_reg = gpuArray(U_reg);
        V_reg = gpuArray(V_reg);
    end

    print(solver, env);

    for k = 1:max_iter

        if (k == 1)
            B = get_embedding_inner(U, V, l, i_idx_R, j_idx_R, env) - R;
            loss = 0.5 * full(sum(sum(B .* B)));
            freq_reg = 0.5 * (sum(U .* U) * U_reg + sum(V .* V) * V_reg);
            f = freq_reg + loss;
            GU = U .* U_reg' + V * B';
            GV = V .* V_reg' + U * B;
            G_norm = norm([GU GV], 'fro');
            G_norm_0 = G_norm;
            fprintf('initial G_norm: %15.6f\n', G_norm_0);
            fprintf('initial reg: %15.6f\n', freq_reg);
            fprintf('initial loss: %15.6f\n', loss);
        end

        if (G_norm <= epsilon * G_norm_0)
            fprintf('Newton stopping condition');
            break;
        end

        time1 = tic;

        if strcmp(solver, 'gauss')
            [U, V, B, loss, f, cg_iters, ls_steps] = update_gauss(U, U_reg, V, V_reg, B, i_idx_R, j_idx_R, loss, f, l, env);
        elseif strcmp(solver, 'alscg')
            GU = U .* U_reg' + V * B';
            [U, B, f, loss, cg_iters_U] = update_block_alscg(U, V, B, l, GU, f, loss, U_reg, 'no_transposed', i_idx_R, j_idx_R, env);
            GV = V .* V_reg' + U * B;
            [V, B, f, loss, cg_iters_V] = update_block_alscg(V, U, B, l, GV, f, loss, V_reg, 'transposed', i_idx_R, j_idx_R, env);
        elseif strcmp(solver, 'als')
            U = updata_block_als(U, V, R, uni_i_idx_R, U_reg, d, m2ns, env);
            V = updata_block_als(V, U, R', uni_j_idx_R, V_reg, d, n2ms, env);
        end

        time2 = toc(time1);
        total_t = total_t + time2;

        if strcmp(solver, 'gauss')
            print_results_gauss(k, total_t, cg_iters, ls_steps, U, V, R_test, i_idx_R_test, j_idx_R_test, nnz_R_test, U_reg, V_reg, B, f, loss, env);
        elseif strcmp(solver, 'alscg')
            print_results_alscg(k, total_t, cg_iters_U, cg_iters_V, U, V, R_test, i_idx_R_test, j_idx_R_test, nnz_R_test, U_reg, V_reg, B, f, loss, env);
        elseif strcmp(solver, 'als')
            print_results_als(k, total_t, U, V, R_test, i_idx_R_test, j_idx_R_test, nnz_R_test, U_reg, V_reg, R, l, i_idx_R, j_idx_R, env)
        end

    end

    if (k == max_iter)
        fprintf('Warning: reach max training iteration. Terminate training process.\n');
    end

end

function [Su, Sv, cg_iters] = cg(U, V, l, G, U_reg, V_reg, i_idx_R, j_idx_R, env)
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
    reg = spdiags([U_reg; V_reg], 0, m + n, m + n);

    while (gamma > eta * eta * gamma_0)
        cg_iters = cg_iters + 1;
        Z = get_cross_embedding_inner(D(:, 1:m), D(:, m + 1:end), U, V, l, i_idx_R, j_idx_R, env);
        Dh = D * reg + [V * Z' U * Z];
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
function Z = get_cross_embedding_inner(Su, Sv, U, V, l, i_idx, j_idx, env)
    vals = zeros(1, l);

    if strcmp(env, 'gpu')
        vals = gpuArray(vals);
    end

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
function Z = get_embedding_inner(U, V, l, i_idx, j_idx, env)
    vals = zeros(1, l);

    if strcmp(env, 'gpu')
        vals = gpuArray(vals);
    end

    num_batches = 10;
    bsize = ceil(l / num_batches);

    for i = 1:num_batches
        range = (i - 1) * bsize + 1:min(l, i * bsize);
        vals(range) = dot(V(:, j_idx(range)), U(:, i_idx(range)));
    end

    Z = sparse(i_idx, j_idx, vals, size(U, 2), size(V, 2));
end

function [] = print(solver, env)
    fprintf('solver: %5s\n', solver);
    fprintf('env: %3s\n', env);

    if strcmp(solver, 'gauss')
        fprintf('%4s  %15s  %3s  %3s  %15s  %15s  %15s  %15s  %15s\n', 'iter', 'time', '#cg', '#ls', 'obj', '|G|', 'test_loss', '|G_U|', '|G_V|', 'loss');
    elseif strcmp(solver, 'alscg')
        fprintf('%4s  %15s  %3s  %3s  %15s  %15s  %15s  %15s  %15s  %15s\n', 'iter', 'time', '#cg_U', '#cg_V', 'obj', '|G_U|', '|G_V|', 'test_loss', '|G|', 'loss');
    elseif strcmp(solver, 'als')
        fprintf('%4s  %15s  %15s  %15s  %15s\n', 'iter', 'time', 'obj', 'test_loss', 'loss');
    end

end

function [U, V, B, loss, f, cg_iters, ls_steps] = update_gauss(U, U_reg, V, V_reg, B, i_idx_R, j_idx_R, loss, f, l, env)
    m = size(U, 2);
    n = size(V, 2);
    nu = 0.1;
    min_step_size = 1e-20;
    G = [U * spdiags(U_reg, 0, m, m) V * spdiags(V_reg, 0, n, n)] + [V * B' U * B];
    [Su, Sv, cg_iters] = cg(U, V, l, G, U_reg, V_reg, i_idx_R, j_idx_R, env);
    Delta_1 = get_cross_embedding_inner(Su, Sv, U, V, l, i_idx_R, j_idx_R, env);
    Delta_2 = get_embedding_inner(Su, Sv, l, i_idx_R, j_idx_R, env);
    US_u = sum(U .* Su) * U_reg; VS_v = sum(V .* Sv) * V_reg;
    SS = sum([Su Sv] .* [Su Sv]) * [U_reg; V_reg];
    GS = sum(sum(G .* [Su Sv]));
    theta = 1;

    for ls_steps = 0:intmax;

        if (theta < min_step_size)
            fprintf('Warning: step size is too small in line search. Switch to the next block of variables.\n');
            return;
        end

        B_new = B + theta * Delta_1 + theta * theta * Delta_2;
        loss_new = 0.5 * full(sum(sum(B_new .* B_new)));
        f_diff = 0.5 * (2 * theta * (US_u + VS_v) + theta * theta * SS) + loss_new - loss;

        if (f_diff <= nu * theta * GS)
            loss = loss_new;
            f = f + f_diff;
            U = U + theta * Su;
            V = V + theta * Sv;
            B = B_new;
            break;
        end

        theta = theta * 0.5;
    end

end

function [U, B, f, loss, cg_iters] = update_block_alscg(U, V, B, l, G, f, loss, reg, option, i_idx_R, j_idx_R, env)
    eta = 0.3;
    cg_max_iter = 20;
    Su = zeros(size(G));
    C = -G;
    D = C;
    gamma_0 = sum(sum(C .* C));
    gamma = gamma_0;
    cg_iters = 0;

    while (gamma > eta * eta * gamma_0)
        cg_iters = cg_iters + 1;

        if strcmp(option, 'transposed')
            Z = get_embedding_inner(V, D, l, i_idx_R, j_idx_R, env);
            Dh = D .* reg' + V * Z;
        else
            Z = get_embedding_inner(D, V, l, i_idx_R, j_idx_R, env);
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
        Delta = get_embedding_inner(V, Su, l, i_idx_R, j_idx_R, env);
    else
        Delta = get_embedding_inner(Su, V, l, i_idx_R, j_idx_R, env);
    end

    B_new = B + Delta;
    USu = sum(U .* Su);
    SuSu = sum(Su .* Su);
    loss_new = 0.5 * full(sum(sum(B_new .* B_new)));
    f_diff = 0.5 * ((2 * USu + SuSu) * reg) + loss_new - loss;
    f = f + f_diff;
    U = U + Su;
    B = B_new;
    loss = loss_new;
end

function U = updata_block_als(U, V, R, uni_i_idx_R, U_reg, d, m2ns, env)
    temp = zeros(d, length(uni_i_idx_R));

    parfor i = 1:length(uni_i_idx_R)
        ii = uni_i_idx_R(i);
        idx = m2ns{ii};
        VVT = (V(:, idx) * V(:, idx)');
        A = VVT + U_reg(ii) * eye(d);
        b = V * R(ii, :)';
        temp(:, i) = A \ b;
    end

    U(:, uni_i_idx_R) = temp;
end

function [] = print_results_gauss(k, total_t, cg_iters, ls_steps, U, V, R_test, i_idx_R_test, j_idx_R_test, nnz_R_test, U_reg, V_reg, B, f, loss, env)
    m = size(U, 2);
    n = size(V, 2);
    Y_test_tilde = get_embedding_inner(U, V, nnz(R_test), i_idx_R_test, j_idx_R_test, env);
    test_loss = sqrt(full(sum(sum((R_test - Y_test_tilde) .* (R_test - Y_test_tilde)))) / nnz_R_test);
    G = [U * spdiags(U_reg, 0, m, m) V * spdiags(V_reg, 0, n, n)] + [V * B' U * B];
    G_norm = norm(G, 'fro');
    GU_norm = norm(G(:, 1:m), 'fro');
    GV_norm = norm(G(:, m + 1:end), 'fro');
    fprintf('%4d  %15.3f  %3d  %3d  %15.3f  %15.6f  %15.6f  %15.6f  %15.6f  %15.3f\n', k, total_t, cg_iters, ls_steps, f, G_norm, test_loss, GU_norm, GV_norm, loss);
end

function [] = print_results_alscg(k, total_t, cg_iters_U, cg_iters_V, U, V, R_test, i_idx_R_test, j_idx_R_test, nnz_R_test, U_reg, V_reg, B, f, loss, env)
    Y_test_tilde = get_embedding_inner(U, V, nnz_R_test, i_idx_R_test, j_idx_R_test, env);
    test_loss = sqrt(full(sum(sum((R_test - Y_test_tilde) .* (R_test - Y_test_tilde)))) / nnz_R_test);
    GU = U .* U_reg' + V * B';
    GV = V .* V_reg' + U * B;
    G_norm = norm([GU GV], 'fro');
    G_norm_U = norm(GU, 'fro');
    G_norm_V = norm(GV, 'fro');
    fprintf('%4d  %15.3f  %3d  %3d  %15.3f  %15.6f  %15.6f  %15.6f  %15.6f  %15.3f\n', k, total_t, cg_iters_U, cg_iters_V, f, G_norm_U, G_norm_V, test_loss, G_norm, loss);
end

function [] = print_results_als(k, total_t, U, V, R_test, i_idx_R_test, j_idx_R_test, nnz_R_test, U_reg, V_reg, R, l, i_idx_R, j_idx_R, env)
    Y_test_tilde = get_embedding_inner(U, V, nnz_R_test, i_idx_R_test, j_idx_R_test, env);
    test_loss = sqrt(full(sum(sum((R_test - Y_test_tilde) .* (R_test - Y_test_tilde)))) / nnz_R_test);
    B = get_embedding_inner(U, V, l, i_idx_R, j_idx_R, env) - R;
    loss = 0.5 * full(sum(sum(B .* B)));
    f = 0.5 * (sum(U .* U) * U_reg + sum(V .* V) * V_reg) + loss;
    fprintf('%4d  %15.3f  %15.3f  %15.6f  %15.3f\n', k, total_t, f, test_loss, loss);
end
