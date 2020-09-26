#include "matrix.h"
#include "mex.h"
#include <omp.h>
#include <stdio.h>

/*
 * get_embedding_inner.c
 *
 * Input:
 * U: a (d-by-m) matrix
 * V: a (d-by-n) matrix
 * i_idx: a (l-by-1) matrix
 * j_idx: a (l-by-1) matrix
 *
 * Output:
 * Z: a (m-by-n) sparse matrix
 *
 * Detailed calculation operation:
 * Z_(m,n) = u_m^T*v_n if (m, n) is equal to (i_idx[k], j_idx[k]) where k is in
 * [0, l-1]
 *
 * The calling syntax is:
 * Z = get_embedding_inner(U, V, i_idx, j_idx)
 *
 * This is a MEX file for MATLAB.
 */

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    /* Check variables */
    // Check whether there are four input variables.
    if (nrhs != 4) {
        mexErrMsgIdAndTxt("MyToolbox:get_embedding_inner:nrhs",
                          "Four inputs required.");
    }

    // Check output number of output variable is one.
    if (nlhs != 1) {
        mexErrMsgIdAndTxt("MyToolbox:get_embedding_inner:nlhs",
                          "One output required.");
    }

    // Check input U, V, i_idx, j_idx are Double arraies.
    for (int i = 0; i < 4; i++) {
        if (!mxIsDouble(prhs[i]) || mxIsComplex(prhs[i])) {
            mexErrMsgIdAndTxt("MyToolbox:get_embedding_inner:notDouble",
                              "Input matrix must be type double.");
        }
    }

    // Check U, V has the same latent dimension.
    if (mxGetM(prhs[0]) != mxGetM(prhs[1])) {
        mexErrMsgIdAndTxt(
            "MyToolbox:get_embedding_inner:dimensionNotMatched",
            "Input U and V matrixies must has the same latent dimension.");
    }

    // Check i_idx, j_idx has the same dimensions.
    if (mxGetM(prhs[2]) != mxGetM(prhs[3]) ||
        mxGetN(prhs[2]) != mxGetN(prhs[3])) {
        mexErrMsgIdAndTxt(
            "MyToolbox:get_embedding_inner:dimensionNotMatched",
            "Input i_idx and j_idx matrixies must has the same dimensions.");
    }

    if (mxGetN(prhs[2]) != 1)
        mexErrMsgIdAndTxt(
            "MyToolbox:get_embedding_inner:wrongDimension",
            "Input i_idx and j_idx matrixies must be (l-by-1) matrix.");

    /* variable declarations here */

    mwSize d, m, n, l;
    d = mxGetM(prhs[0]);
    m = mxGetN(prhs[0]);
    n = mxGetN(prhs[1]);
    l = mxGetM(prhs[3]);

    mxDouble *U = mxGetPr(prhs[0]);
    mxDouble *V = mxGetPr(prhs[1]);
    mxDouble *i_idx = mxGetPr(prhs[2]);
    mxDouble *j_idx = mxGetPr(prhs[3]);

    /* code here */
    plhs[0] = mxCreateSparse(m, n, l, 0);
    mwIndex *jr = mxGetIr(plhs[0]);
    mwIndex *jc = mxGetJc(plhs[0]);
    mxDouble *pr = mxGetPr(plhs[0]);

    omp_set_num_threads(5);

    jc[0] = 0;
    int cur_col_idx = 0;
    for (int i = 0; i < l; i++) {
        jr[i] = (i_idx[i] - 1);

        while (cur_col_idx <= (j_idx[i] - 1)) {
            jc[cur_col_idx] = i;
            cur_col_idx++;
        }
    }

    while (cur_col_idx <= n) {
        jc[cur_col_idx] = l;
        cur_col_idx++;
    }

#pragma omp parallel for schedule(guided)
    for (int i = 0; i < l; i++) {
        for (int k = 0; k < d; k++) {
            pr[i] += U[(int)((i_idx[i] - 1) * d + k)] *
                     V[(int)((j_idx[i] - 1) * d + k)];
        }
    }
}
