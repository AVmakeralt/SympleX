// SympleX Kernel Library — compiled to shared library
// All kernels use cache-friendly loop orders and -O3 -march=native

#include <stddef.h>
#include <stdint.h>

void symplex_matmul_f32(const float* A, const float* B, float* C,
                         int64_t M, int64_t N, int64_t K) {
    for (int64_t i = 0; i < M; i++)
        for (int64_t j = 0; j < N; j++)
            C[i * N + j] = 0.0f;
    for (int64_t i = 0; i < M; i++)
        for (int64_t k = 0; k < K; k++) {
            float a_ik = A[i * K + k];
            for (int64_t j = 0; j < N; j++)
                C[i * N + j] += a_ik * B[k * N + j];
        }
}

void symplex_add_f32(float* dst, const float* a, const float* b, int64_t n) {
    for (int64_t i = 0; i < n; i++) dst[i] = a[i] + b[i];
}

void symplex_mul_f32(float* dst, const float* a, const float* b, int64_t n) {
    for (int64_t i = 0; i < n; i++) dst[i] = a[i] * b[i];
}

void symplex_sub_f32(float* dst, const float* a, const float* b, int64_t n) {
    for (int64_t i = 0; i < n; i++) dst[i] = a[i] - b[i];
}

void symplex_stencil_2d(float* out, const float* in, int64_t N, float dx) {
    float inv_dx2 = 1.0f / (dx * dx);
    for (int64_t i = 1; i < N - 1; i++)
        for (int64_t j = 1; j < N - 1; j++)
            out[i * N + j] = (in[(i-1)*N+j] + in[(i+1)*N+j] + in[i*N+(j-1)] + in[i*N+(j+1)] - 4.0f*in[i*N+j]) * inv_dx2;
}

void symplex_nbody_forces(const float* pos_x, const float* pos_y,
                           const float* mass, float* force_x, float* force_y,
                           int64_t n, float G, float softening) {
    for (int64_t i = 0; i < n; i++) {
        float fx = 0.0f, fy = 0.0f;
        for (int64_t j = 0; j < n; j++) {
            if (i == j) continue;
            float dx = pos_x[j] - pos_x[i];
            float dy = pos_y[j] - pos_y[i];
            float r2 = dx*dx + dy*dy + softening*softening;
            float f = G * mass[i] * mass[j] / r2;
            fx += f * dx; fy += f * dy;
        }
        force_x[i] = fx; force_y[i] = fy;
    }
}

void symplex_euler_step(float* pos, float* vel, const float* force,
                         float mass, float dt, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        vel[i] += (force[i] / mass) * dt;
        pos[i] += vel[i] * dt;
    }
}

void symplex_rk4_step(float* x, float* v, float k, float c, float m,
                       float dt, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        float xi = x[i], vi = v[i];
        float k1x = vi, k1v = (-k*xi - c*vi)/m;
        float k2x = vi + 0.5f*dt*k1v, k2v = (-k*(xi+0.5f*dt*k1x) - c*k2x)/m;
        float k3x = vi + 0.5f*dt*k2v, k3v = (-k*(xi+0.5f*dt*k2x) - c*k3x)/m;
        float k4x = vi + dt*k3v, k4v = (-k*(xi+dt*k3x) - c*k4x)/m;
        x[i] = xi + (dt/6.0f)*(k1x + 2.0f*k2x + 2.0f*k3x + k4x);
        v[i] = vi + (dt/6.0f)*(k1v + 2.0f*k2v + 2.0f*k3v + k4v);
    }
}

void symplex_grad_magnitude(float* out, const float* gx, const float* gy, int64_t n) {
    for (int64_t i = 0; i < n; i++)
        out[i] = gx[i]*gx[i] + gy[i]*gy[i];
}

void symplex_matmul_tiled_f32(const float* A, const float* B, float* C,
                               int64_t M, int64_t N, int64_t K) {
    for (int64_t i = 0; i < M * N; i++) C[i] = 0.0f;
    const int64_t TM = 64, TN = 64, TK = 64;
    for (int64_t ti = 0; ti < M; ti += TM) {
        int64_t mi = (ti + TM < M) ? TM : (M - ti);
        for (int64_t tj = 0; tj < N; tj += TN) {
            int64_t mj = (tj + TN < N) ? TN : (N - tj);
            for (int64_t tk = 0; tk < K; tk += TK) {
                int64_t mk = (tk + TK < K) ? TK : (K - tk);
                for (int64_t i = 0; i < mi; i++)
                    for (int64_t k = 0; k < mk; k++) {
                        float a_ik = A[(ti+i)*K + (tk+k)];
                        for (int64_t j = 0; j < mj; j++)
                            C[(ti+i)*N + (tj+j)] += a_ik * B[(tk+k)*N + (tj+j)];
                    }
            }
        }
    }
}
