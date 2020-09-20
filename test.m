% set model parameters
lambda_U = 0.5; lambda_V = 0.5; d = 8;
tr = 'trva'; va = 'va';

% set training algorithm's parameters
epsilon = 1e-5;
max_iter = 10;

% prepare training and test data sets
R = mf_read(tr);
R_test = mf_read(va);

m = max(size(R, 1), size(R_test, 1));
n = max(size(R, 2), size(R_test, 2));

[i, j, s] = find(R);
R = sparse(i, j, s, m, n);
[i, j, s] = find(R_test);
R_test = sparse(i, j, s, m, n);

%Init freq regularization
IR = spones(R);
U_reg = full(sum(IR')' * lambda_U);
V_reg = full(sum(IR)' * lambda_V);

% learn an FM model
rand('seed', 0);
U = 2 * (0.1 / sqrt(d)) * (rand(d, m) - 0.5);
V = 2 * (0.1 / sqrt(d)) * (rand(d, n) - 0.5);

solver = 'alscg';
env = 'cpu';

[U, V] = mf_train(R, U, V, U_reg, V_reg, epsilon, max_iter, R_test, solver, env);

% do prediction
%y_tilde = fm_predict(X_test, w, U, V);
%display(sprintf('test accuracy: %f', sum(sign(y_tilde) == y_test)/size(y_test,1)));
