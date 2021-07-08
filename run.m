function run(solver, enable_gpu, l2, d, t, eta, cgt)
  % function run(solver, enable_gpu, l2, d, t)
  % Inputs:
  % solver: 
  %   - 0: gauss
  %   - 1: alscg
  % enable_gpu: 
  %   - 0: disable
  %   - 1: enable
  % l2: regularization
  % d: embedding dimemsion
  % t: iteration
  % eta: cg tightness
  % cgt: max cg iteration

  % Set terminate condition 
  epsilon = 1e-6;

  % Prepare training and test data sets
  R = mf_read('tr');
  R_test = mf_read('va');

  m = max(size(R, 1), size(R_test, 1));
  n = max(size(R, 2), size(R_test, 2));

  [i, j, s] = find(R);
  R = sparse(i, j, s, m, n);
  [i, j, s] = find(R_test);
  R_test = sparse(i, j, s, m, n);

  % Init freq regularization
  IR = spones(R);
  U_reg = full(sum(IR')' * l2);
  V_reg = full(sum(IR)' * l2);

  % Load model
  U = readmatrix('P.model', 'FileType', 'text');
  V = readmatrix('Q.model', 'FileType', 'text');

  % Choose solver
  if (solver == 0) 
    solver = 'gauss';
  elseif (solver == 1)
    solver = 'alscg';
  end

  % Enable GPU
  if (enable_gpu)
    env = 'gpu';
  else
    env = 'cpu';

  [U, V] = mf_train(R, U', V', U_reg, V_reg, epsilon, t, R_test, solver, env, eta, cgt);
end
