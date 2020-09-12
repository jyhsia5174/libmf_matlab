function R = mf_read(tr)
    R = load(tr);
    R = sparse(R(:, 1), R(:, 2), R(:, 3));
end
