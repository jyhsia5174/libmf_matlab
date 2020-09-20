d = 40;
m = 10000;
n = m / 100;

U = rand(d, m);
V = rand(d, n);
R = sprand(m, n, 10 / n);
[i_idx, j_idx, vals] = find(R);

tic;
Z = get_embedding_inner(U, V, i_idx, j_idx);
toc

%tic;
%Ztruth = get_embedding_inner_truth(U, V, i_idx, j_idx);
%toc

U = gpuArray(U);
V = gpuArray(V);
tic;
Zgpu = get_embedding_inner_truth_gpu(U, V, i_idx, j_idx);
toc

%norm(Z - Ztruth, 'fro')

function Z = get_embedding_inner_truth(U, V, i_idx, j_idx)
    l = size(i_idx, 1);
    vals = zeros(1, l);

    num_batches = 10;
    bsize = ceil(l / num_batches);

    for i = 1:num_batches
        range = (i - 1) * bsize + 1:min(l, i * bsize);
        vals(range) = dot(V(:, j_idx(range)), U(:, i_idx(range)));
    end

    Z = sparse(i_idx, j_idx, vals, size(U, 2), size(V, 2));
end

function Z = get_embedding_inner_truth_gpu(U, V, i_idx, j_idx)
    l = size(i_idx, 1);
    vals = zeros(1, l);

    vals = gpuArray(vals);

    num_batches = 10;
    bsize = ceil(l / num_batches);

    for i = 1:num_batches
        range = (i - 1) * bsize + 1:min(l, i * bsize);
        vals(range) = dot(V(:, j_idx(range)), U(:, i_idx(range)));
    end

    Z = sparse(i_idx, j_idx, vals, size(U, 2), size(V, 2));
end
