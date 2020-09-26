#include "matrix.h"
#include "mex.h"
#include <omp.h>
#include <stdio.h>

/*
 * get_embedding_inner.c
 *
 * Input:
 * Su: a (d-by-m) matrix
 * Sv: a (d-by-n) matrix
 * U: a (d-by-m) matrix
 * V: a (d-by-n) matrix
 * i_idx: a (l-by-1) matrix
 * j_idx: a (l-by-1) matrix
 *
 * Output:
 * D: a (m-by-n) sparse matrix
 *
 * Detailed calculation operation:
 * D_(m,n) = u_m^T*v_n if (m, n) is equal to (i_idx[k], j_idx[k]) where k is in
 * [0, l-1]
 *
 * The calling syntax is:
 * D = get_cross_embedding_inner(U, V, i_idx, j_idx)
 *
 * This is a MEX file for MATLAB.
 */

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    /* Check variables */
    // Check whether there are four input variables.
    if (nrhs != 6) {
        mexErrMsgIdAndTxt("MyToolbox:get_cross_embedding_inner:nrhs",
                          "Six inputs required.");
    }

    // Check output number of output variable is one.
    if (nlhs != 1) {
        mexErrMsgIdAndTxt("MyToolbox:get_cross_embedding_inner:nlhs",
                          "One output required.");
    }

    // Check input U, V, i_idx, j_idx are Double arraies.
    for (int i = 0; i < 6; i++) {
        if (!mxIsDouble(prhs[i]) || mxIsComplex(prhs[i])) {
            mexErrMsgIdAndTxt("MyToolbox:get_cross_embedding_inner:notDouble",
                              "Input matrix must be type double.");
        }
    }

    // Check Su, U has the same latent dimension.
    if (mxGetM(prhs[0]) != mxGetM(prhs[2]) ||
        mxGetN(prhs[0]) != mxGetN(prhs[2])) {
        mexErrMsgIdAndTxt(
            "MyToolbox:get_cross_embedding_inner:dimensionNotMatched",
            "Input Su and U matrixies must has the same dimension.");
    }

    // Check Sv, V has the same latent dimension.
    if (mxGetM(prhs[1]) != mxGetM(prhs[3]) ||
        mxGetN(prhs[1]) != mxGetN(prhs[3])) {
        mexErrMsgIdAndTxt(
            "MyToolbox:get_cross_embedding_inner:dimensionNotMatched",
            "Input Sv and V matrixies must has the same dimension.");
    }

    if (mxGetM(prhs[0]) != mxGetM(prhs[1]) ||
        mxGetM(prhs[1]) != mxGetM(prhs[2]) ||
        mxGetM(prhs[2]) != mxGetM(prhs[3])) {
        mexErrMsgIdAndTxt(
            "MyToolbox:get_cross_embedding_inner:dimensionNotMatched",
            "Input Sv, Sv, U and V matrixies must has the same latent "
            "dimension.");
    }

    // Check i_idx, j_idx has the same dimensions.
    if (mxGetM(prhs[4]) != mxGetM(prhs[5]) ||
        mxGetN(prhs[4]) != mxGetN(prhs[5])) {
        mexErrMsgIdAndTxt(
            "MyToolbox:get_cross_embedding_inner:dimensionNotMatched",
            "Input i_idx and j_idx matrixies must has the same dimensions.");
    }

    if (mxGetN(prhs[4]) != 1 || mxGetN(prhs[5]) != 1)
        mexErrMsgIdAndTxt(
            "MyToolbox:get_embedding_inner:wrongDimension",
            "Input i_idx and j_idx matrixies must be (L-by-1) matrix.");

    /* variable declarations here */

    mwSize d, m, n, l;
    d = mxGetM(prhs[0]);
    m = mxGetN(prhs[0]);
    n = mxGetN(prhs[1]);
    l = mxGetM(prhs[5]);

    mxDouble *Su = mxGetPr(prhs[0]);
    mxDouble *Sv = mxGetPr(prhs[1]);
    mxDouble *U = mxGetPr(prhs[2]);
    mxDouble *V = mxGetPr(prhs[3]);
    mxDouble *i_idx = mxGetPr(prhs[4]);
    mxDouble *j_idx = mxGetPr(prhs[5]);

    /* code here */
    plhs[0] = mxCreateSparse(m, n, l, 0);
    mwIndex *jr = mxGetIr(plhs[0]);
    mwIndex *jc = mxGetJc(plhs[0]);
    mxDouble *pr = mxGetPr(plhs[0]);

    omp_set_num_threads(12);

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
            pr[i] += Su[(int)((i_idx[i] - 1) * d + k)] *
                         V[(int)((j_idx[i] - 1) * d + k)] +
                     U[(int)((i_idx[i] - 1) * d + k)] *
                         Sv[(int)((j_idx[i] - 1) * d + k)];
        }
    }
}
