#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fftw3.h>
#include <time.h>

#define N 16384

void init_f(double* p, int power, bool powerOf2) {
    for (int i = 0; i < N; i++) p[i] = 0.0;
    if (powerOf2) {
        p[0] = 1.0;
        p[power / 2] = 1.0;
    }
    else {
        for (int i = 0; i < N; i++) p[i] = (i < power) ? 1.0 : 0.0;
    }
}


void multiply_polys(double* a, double* b, double* result) {
    fftw_complex* A = fftw_alloc_complex(N);
    fftw_complex* B = fftw_alloc_complex(N);
    fftw_complex* C = fftw_alloc_complex(N);

    fftw_plan p1 = fftw_plan_dft_r2c_1d(N, a, A, FFTW_ESTIMATE);
    fftw_plan p2 = fftw_plan_dft_r2c_1d(N, b, B, FFTW_ESTIMATE);
    fftw_execute(p1);
    fftw_execute(p2);
    fftw_destroy_plan(p1);
    fftw_destroy_plan(p2);

    for (int i = 0; i < N; i++) {
        double reA = A[i][0], imA = A[i][1];
        double reB = B[i][0], imB = B[i][1];
        C[i][0] = reA * reB - imA * imB;
        C[i][1] = reA * imB + imA * reB;
    }

    fftw_plan pinv = fftw_plan_dft_c2r_1d(N, C, result, FFTW_ESTIMATE);
    fftw_execute(pinv);
    fftw_destroy_plan(pinv);
    for (int i = 0; i < N; i++) result[i] /= N;

    fftw_free(A);
    fftw_free(B);
    fftw_free(C);
    fftw_cleanup();
}

// f(x) = f1 * f2 * f3
void compute_f(double* f, int q0, int q1, int q2) {
    double* f1 = fftw_alloc_real(N);
    double* f2 = fftw_alloc_real(N);
    double* f3 = fftw_alloc_real(N);
    fftw_complex* F1 = fftw_alloc_complex(N);
    fftw_complex* F2 = fftw_alloc_complex(N);
    fftw_complex* F3 = fftw_alloc_complex(N);
    fftw_complex* F = fftw_alloc_complex(N);

    init_f(f1, q0, true);
    init_f(f2, q1, false);
    init_f(f3, q2, false);

    fftw_plan p1 = fftw_plan_dft_r2c_1d(N, f1, F1, FFTW_ESTIMATE);
    fftw_plan p2 = fftw_plan_dft_r2c_1d(N, f2, F2, FFTW_ESTIMATE);
    fftw_plan p3 = fftw_plan_dft_r2c_1d(N, f3, F3, FFTW_ESTIMATE);
    fftw_execute(p1);
    fftw_execute(p2);
    fftw_execute(p3);
    fftw_destroy_plan(p1);
    fftw_destroy_plan(p2);
    fftw_destroy_plan(p3);

    for (int i = 0; i < N; i++) {
        double re1 = F1[i][0], im1 = F1[i][1];
        double re2 = F2[i][0], im2 = F2[i][1];
        double re3 = F3[i][0], im3 = F3[i][1];
        double temp_re, temp_im;

        temp_re = re1 * re2 - im1 * im2;
        temp_im = re1 * im2 + im1 * re2;
        F[i][0] = temp_re * re3 - temp_im * im3;
        F[i][1] = temp_re * im3 + temp_im * re3;
    }

    fftw_plan pinv = fftw_plan_dft_c2r_1d(N, F, f, FFTW_ESTIMATE);
    fftw_execute(pinv);
    fftw_destroy_plan(pinv);
    for (int i = 0; i < N; i++) f[i] /= N;

    fftw_free(f1);
    fftw_free(f2);
    fftw_free(f3);
    fftw_free(F1);
    fftw_free(F2);
    fftw_free(F3);
    fftw_free(F);
    fftw_cleanup();
}

void poly_mod(double* a, double* f, int n) {
    int deg_f = n - 1;
    while (deg_f >= 0 && fabs(f[deg_f]) < 1e-10) deg_f--;

    if (deg_f < 0) {
        printf("Illegal modular polynomial\n");
        exit(1);
    }

    while (1) {
        int deg_a = n - 1;
        while (deg_a >= 0 && fabs(a[deg_a]) < 1e-10) deg_a--;
        if (deg_a < deg_f) break;

        double factor = a[deg_a] / f[deg_f];
        int shift = deg_a - deg_f;

        for (int i = 0; i <= deg_f; i++) {
            a[shift + i] -= factor * f[i];
        }
    }
}

void poly_mul(double* a, double* b, double* c, int n) {

    for (int i = 0; i < 2 * n - 1; i++) c[i] = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            c[i + j] += a[i] * b[j];
        }
    }
}

void poly_mod2(double* c, double* f, int max_deg_c, int deg_f) {

    for (int i = max_deg_c; i >= deg_f; i--) {
        if (fabs(c[i]) < 1e-10) continue;

        double factor = c[i] / f[deg_f];
        for (int j = 0; j <= deg_f; j++) {
            c[i - j] -= factor * f[deg_f - j];
        }
    }
}

void random_poly(double* p, int n, int max_coeff, int density) {
    for (int i = 0; i < n; i++) {
        if (rand() % density == 0) {
            p[i] = (double)(rand() % max_coeff + 1); 
        }
        else {
            p[i] = 0.0;
        }
    }
}

int main() {
    int q0 = 4, q1 = 41, q2 = 73;
    int degf = 114;
    double* a = fftw_alloc_real(N);
    double* b = fftw_alloc_real(N);
    double* c = fftw_alloc_real(N);
    double* d = fftw_alloc_real(2 * N);
    double* f = fftw_alloc_real(N);

    random_poly(a, degf, 10, 3);
    random_poly(b, degf, 10, 3);

    // Step 1: f(x) = f1 * f2 * f3
    compute_f(f, q0, q1, q2);

    // Step 2: c = fftw(a * b)
    multiply_polys(a, b, c);

    // Step 3: c mod f(x)
    poly_mod(c, f, N);

    // Step 4: d = a * b
    poly_mul(a, b, d, N);

    // Step 5: d mod f(x)
    poly_mod2(d, f, 2 * N - 1, degf);

    // Step 6: Verify
    printf("Coefficients of two results:\n");
    for (int i = 0; i < N; i++) {
        if (fabs(c[i]) > 1e-10) {
            printf("x^%d fftw: %.6f, mult: %.6f", i, c[i], d[i]);
            if (int(fabs(c[i])) == int(fabs(d[i]))) printf("   equal\n");
            else printf("    not equal\n");
        }
    }

    // Clear
    fftw_free(a);
    fftw_free(b);
    fftw_free(c);
    fftw_free(d);
    fftw_free(f);
    fftw_cleanup();


    return 0;
}