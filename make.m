% This make.m is for MATLAB and OCTAVE under Windows, Mac, and Unix
try
    mex CFLAGS="\$CFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp" get_embedding_inner_cpu.c
    mex CFLAGS="\$CFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp" get_cross_embedding_inner_cpu.c
catch
    fprintf('If make.m fails, please check README about detailed instructions.\n');
end
