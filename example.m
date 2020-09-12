% set model parameters
%lambda_U = 1e-7; lambda_V = 1e-7; d = 4;
%lambda_U = 0.05; lambda_V = 0.05; d = 20;
%tr = 'tr'; va = 'te';

% set training algorithm's parameters
%epsilon = 1e-6;
%epsilon = 1e-5;
%max_iter = 300;

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
%rand('seed', 0);
%U = 2*(0.1/sqrt(d))*(rand(d,m)-0.5);
%V = 2*(0.1/sqrt(d))*(rand(d,n)-0.5);

U = dlmread('initial_model_P');
V = dlmread('initial_model_Q');

%solver = 'als';
%env = 'cpu';

[U, V] = mf_train(R, U', V', U_reg, V_reg, epsilon, max_iter, R_test, d, solver, env);
% do prediction
%y_tilde = fm_predict(X_test, w, U, V);
%display(sprintf('test accuracy: %f', sum(sign(y_tilde) == y_test)/size(y_test,1)));
