// SympleX Physics Benchmark Suite
//
// Benchmarks SympleX's JIT-compiled kernels (SSE2 scalar) against pure Rust
// AVX2/FMA implementations across 5 physics workload categories:
//   1. PDE Stencils (heat & wave equations)
//   2. N-body simulation
//   3. Linear algebra chains (fused matmul)
//   4. Physics integrators (Euler, RK4)
//   5. Tensor field operations (gradient, divergence, curl, laplacian)
//
// Run: cargo bench --bench physics_bench

use std::time::Instant;

// ── SympleX JIT API ──────────────────────────────────────────────────────────
use symplex_poly::{
    SimdLevel, detect_simd_level,
    compile_matmul,
    JitKernel,
};

// ══════════════════════════════════════════════════════════════════════════════
// Utilities
// ══════════════════════════════════════════════════════════════════════════════

struct BenchResult {
    name: String,
    time_ms: f64,
    gflops: f64,
}

fn print_section(title: &str) {
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  {}", title);
    println!("═══════════════════════════════════════════════════════════════════");
}

fn print_comparison_table(results: &[BenchResult], numpy_time_ms: f64, numpy_gflops: f64) {
    println!("┌──────────────────────────┬────────────┬───────────┬──────────────┐");
    println!("│ Implementation           │   Time(ms) │   GFLOPS  │ Speedup vsNP │");
    println!("├──────────────────────────┼────────────┼───────────┼──────────────┤");
    for r in results {
        let speedup = if numpy_time_ms > 0.0 { numpy_time_ms / r.time_ms } else { 0.0 };
        println!("│ {:<24} │ {:>10.3} │ {:>9.3} │ {:>12.2}x │",
                 r.name, r.time_ms, r.gflops, speedup);
    }
    println!("│ {:<24} │ {:>10.3} │ {:>9.3} │ {:>12.2}x │",
             "NumPy (reference)", numpy_time_ms, numpy_gflops, 1.0);
    println!("└──────────────────────────┴────────────┴───────────┴──────────────┘");
}

/// Warm up the CPU by spinning for a few ms
fn warmup() {
    let start = Instant::now();
    let mut x: f64 = 1.0;
    while start.elapsed().as_millis() < 50 {
        x = x * 1.0000001 + 0.0000001;
    }
    std::hint::black_box(x);
}

/// Run a closure multiple times, return median time in ms
fn bench_closure<F: FnMut()>(mut f: F, warmup_iters: usize, bench_iters: usize) -> f64 {
    // Warmup
    for _ in 0..warmup_iters {
        f();
    }
    // Measure
    let mut times = Vec::with_capacity(bench_iters);
    for _ in 0..bench_iters {
        let t0 = Instant::now();
        f();
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[times.len() / 2]
}

/// Call a JIT kernel with the given slot arrays
unsafe fn call_jit_kernel(
    kernel: &JitKernel,
    slot_ptrs: &[*mut f64],
    slot_sizes: &[i64],
) -> i64 {
    kernel.call(slot_ptrs, slot_sizes)
}

// ══════════════════════════════════════════════════════════════════════════════
// 1. PDE Stencils
// ══════════════════════════════════════════════════════════════════════════════

fn bench_pde_stencils() {
    print_section("1. PDE Stencils — Heat & Wave Equations (256×256, 100 steps)");

    let nx: usize = 256;
    let ny: usize = 256;
    let n = nx * ny;
    let steps = 100usize;

    let alpha = 0.01f64;
    let dt = 0.1f64;
    let c = 1.0f64;
    let coeff_heat = alpha * dt; // for heat equation
    let coeff_wave = c * c * dt * dt; // for wave equation

    // ── Scalar Rust (reference implementation) ───────────────────────────
    let mut u_heat = vec![0.0f64; n];
    let u_heat_new = vec![0.0f64; n];
    // Initialize with a Gaussian bump in the center
    for i in 0..nx {
        for j in 0..ny {
            let dx = i as f64 - nx as f64 / 2.0;
            let dy = j as f64 - ny as f64 / 2.0;
            u_heat[i * ny + j] = (-0.5 * (dx * dx + dy * dy) / 100.0).exp();
        }
    }

    let heat_scalar_time = {
        let mut u = u_heat.clone();
        let mut u_new = u_heat_new.clone();
        let time_ms = bench_closure(|| {
            for _ in 0..steps {
                for i in 1..nx-1 {
                    for j in 1..ny-1 {
                        let idx = i * ny + j;
                        u_new[idx] = u[idx] + coeff_heat * (
                            u[(i-1)*ny+j] + u[(i+1)*ny+j] +
                            u[i*ny+(j-1)] + u[i*ny+(j+1)] - 4.0*u[idx]
                        );
                    }
                }
                std::mem::swap(&mut u, &mut u_new);
            }
        }, 3, 15);

        time_ms
    };

    // ── AVX2 Rust implementation of heat equation ────────────────────────
    #[cfg(target_arch = "x86_64")]
    {
        let heat_avx2_time = {
            let mut u = u_heat.clone();
            let mut u_new = u_heat_new.clone();

            let time_ms = bench_closure(|| {
                for _ in 0..steps {
                    unsafe {
                        heat_stencil_avx2(&u, &mut u_new, nx, ny, coeff_heat);
                    }
                    std::mem::swap(&mut u, &mut u_new);
                }
            }, 3, 15);

            time_ms
        };

        let total_flops = 10.0 * (nx - 2) as f64 * (ny - 2) as f64 * steps as f64;
        let scalar_gflops = total_flops / (heat_scalar_time * 1e-3) / 1e9;
        let avx2_gflops = total_flops / (heat_avx2_time * 1e-3) / 1e9;

        println!("\n  Heat Equation: u_new[i,j] = u[i,j] + α·dt·(u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1] - 4·u[i,j])");
        let results = vec![
            BenchResult { name: "Rust scalar".into(), time_ms: heat_scalar_time, gflops: scalar_gflops },
            BenchResult { name: "Rust AVX2/FMA".into(), time_ms: heat_avx2_time, gflops: avx2_gflops },
        ];
        print_comparison_table(&results, 0.0, 0.0); // NumPy time filled by Python script
    }

    // ── Wave equation ────────────────────────────────────────────────────
    let mut u_wave = vec![0.0f64; n];
    let mut u_wave_old = vec![0.0f64; n];
    let u_wave_new = vec![0.0f64; n];
    for i in 0..nx {
        for j in 0..ny {
            let dx = i as f64 - nx as f64 / 2.0;
            let dy = j as f64 - ny as f64 / 2.0;
            u_wave[i * ny + j] = (-0.5 * (dx * dx + dy * dy) / 100.0).exp();
            u_wave_old[i * ny + j] = u_wave[i * ny + j];
        }
    }

    let wave_scalar_time = {
        let mut u = u_wave.clone();
        let mut u_old = u_wave_old.clone();
        let mut u_new = u_wave_new.clone();

        let time_ms = bench_closure(|| {
            for _ in 0..steps {
                for i in 1..nx-1 {
                    for j in 1..ny-1 {
                        let idx = i * ny + j;
                        u_new[idx] = 2.0*u[idx] - u_old[idx] + coeff_wave * (
                            u[(i-1)*ny+j] + u[(i+1)*ny+j] +
                            u[i*ny+(j-1)] + u[i*ny+(j+1)] - 4.0*u[idx]
                        );
                    }
                }
                std::mem::swap(&mut u_old, &mut u);
                std::mem::swap(&mut u, &mut u_new);
            }
        }, 3, 15);

        time_ms
    };

    #[cfg(target_arch = "x86_64")]
    {
        let wave_avx2_time = {
            let mut u = u_wave.clone();
            let mut u_old = u_wave_old.clone();
            let mut u_new = u_wave_new.clone();

            let time_ms = bench_closure(|| {
                for _ in 0..steps {
                    unsafe {
                        wave_stencil_avx2(&u, &u_old, &mut u_new, nx, ny, coeff_wave);
                    }
                    std::mem::swap(&mut u_old, &mut u);
                    std::mem::swap(&mut u, &mut u_new);
                }
            }, 3, 15);

            time_ms
        };

        // Wave eq: ~12 flops per interior point (2* + 4 add + mul + sub + neg + stencil)
        let total_flops = 12.0 * (nx - 2) as f64 * (ny - 2) as f64 * steps as f64;
        let scalar_gflops = total_flops / (wave_scalar_time * 1e-3) / 1e9;
        let avx2_gflops = total_flops / (wave_avx2_time * 1e-3) / 1e9;

        println!("\n  Wave Equation: u_new[i,j] = 2·u[i,j] - u_old[i,j] + c²·dt²·(stencil)");
        let results = vec![
            BenchResult { name: "Rust scalar".into(), time_ms: wave_scalar_time, gflops: scalar_gflops },
            BenchResult { name: "Rust AVX2/FMA".into(), time_ms: wave_avx2_time, gflops: avx2_gflops },
        ];
        print_comparison_table(&results, 0.0, 0.0);
    }
}

/// AVX2 heat stencil: processes 4 doubles at a time per row
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
unsafe fn heat_stencil_avx2(
    u: &[f64], u_new: &mut [f64], nx: usize, ny: usize, coeff: f64,
) {
    use std::arch::x86_64::*;
    let coeff_v = _mm256_set1_pd(coeff);
    let neg4_v = _mm256_set1_pd(-4.0);

    for i in 1..nx-1 {
        let row_off = i * ny;
        let row_up = (i - 1) * ny;
        let row_dn = (i + 1) * ny;

        let j = 1;
        // Process 4 elements at a time
        let mut jj = j;
        while jj + 4 <= ny - 1 {
            let center = _mm256_loadu_pd(u.as_ptr().add(row_off + jj));
            let up     = _mm256_loadu_pd(u.as_ptr().add(row_up + jj));
            let down   = _mm256_loadu_pd(u.as_ptr().add(row_dn + jj));
            let left   = _mm256_loadu_pd(u.as_ptr().add(row_off + jj - 1));
            let right  = _mm256_loadu_pd(u.as_ptr().add(row_off + jj + 1));

            // stencil = up + down + left + right - 4*center
            let stencil = _mm256_fmadd_pd(neg4_v, center, _mm256_add_pd(
                _mm256_add_pd(up, down),
                _mm256_add_pd(left, right),
            ));
            // u_new = center + coeff * stencil
            let result = _mm256_fmadd_pd(coeff_v, stencil, center);
            _mm256_storeu_pd(u_new.as_mut_ptr().add(row_off + jj), result);
            jj += 4;
        }
        // Handle remaining elements scalar
        while jj < ny - 1 {
            let idx = row_off + jj;
            let stencil = u[row_up+jj] + u[row_dn+jj] + u[idx-1] + u[idx+1] - 4.0*u[idx];
            u_new[idx] = u[idx] + coeff * stencil;
            jj += 1;
        }
    }
}

/// AVX2 wave stencil
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
unsafe fn wave_stencil_avx2(
    u: &[f64], u_old: &[f64], u_new: &mut [f64], nx: usize, ny: usize, coeff: f64,
) {
    use std::arch::x86_64::*;
    let coeff_v = _mm256_set1_pd(coeff);
    let two_v = _mm256_set1_pd(2.0);
    let neg4_v = _mm256_set1_pd(-4.0);

    for i in 1..nx-1 {
        let row_off = i * ny;
        let row_up = (i - 1) * ny;
        let row_dn = (i + 1) * ny;

        let mut jj = 1;
        while jj + 4 <= ny - 1 {
            let center = _mm256_loadu_pd(u.as_ptr().add(row_off + jj));
            let old    = _mm256_loadu_pd(u_old.as_ptr().add(row_off + jj));
            let up     = _mm256_loadu_pd(u.as_ptr().add(row_up + jj));
            let down   = _mm256_loadu_pd(u.as_ptr().add(row_dn + jj));
            let left   = _mm256_loadu_pd(u.as_ptr().add(row_off + jj - 1));
            let right  = _mm256_loadu_pd(u.as_ptr().add(row_off + jj + 1));

            let stencil = _mm256_fmadd_pd(neg4_v, center, _mm256_add_pd(
                _mm256_add_pd(up, down),
                _mm256_add_pd(left, right),
            ));
            // u_new = 2*center - old + coeff * stencil
            let two_c = _mm256_mul_pd(two_v, center);
            let wave = _mm256_fmadd_pd(coeff_v, stencil, _mm256_sub_pd(two_c, old));
            _mm256_storeu_pd(u_new.as_mut_ptr().add(row_off + jj), wave);
            jj += 4;
        }
        while jj < ny - 1 {
            let idx = row_off + jj;
            let stencil = u[row_up+jj] + u[row_dn+jj] + u[idx-1] + u[idx+1] - 4.0*u[idx];
            u_new[idx] = 2.0*u[idx] - u_old[idx] + coeff * stencil;
            jj += 1;
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// 2. N-body Simulation
// ══════════════════════════════════════════════════════════════════════════════

fn bench_nbody() {
    print_section("2. N-body Simulation (N=500, 10 steps, brute-force O(N²))");

    let n_particles: usize = 500;
    let n_steps: usize = 10;
    let g = 6.674e-11;
    let epsilon = 0.01f64;
    let dt = 0.001f64;

    // Initialize particles
    let mut pos_x = Vec::with_capacity(n_particles);
    let mut pos_y = Vec::with_capacity(n_particles);
    let mut pos_z = Vec::with_capacity(n_particles);
    let vel_x = vec![0.0f64; n_particles];
    let vel_y = vec![0.0f64; n_particles];
    let vel_z = vec![0.0f64; n_particles];
    let mass: Vec<f64> = (0..n_particles).map(|i| 1e6 + (i as f64) * 100.0).collect();

    for i in 0..n_particles {
        let angle = 2.0 * std::f64::consts::PI * i as f64 / n_particles as f64;
        let r = 100.0;
        pos_x.push(r * angle.cos());
        pos_y.push(r * angle.sin());
        pos_z.push((i as f64 * 0.1).sin() * 10.0);
    }

    // Scalar N-body
    let nbody_scalar_time = {
        let mut px = pos_x.clone();
        let mut py = pos_y.clone();
        let mut pz = pos_z.clone();
        let mut vx = vel_x.clone();
        let mut vy = vel_y.clone();
        let mut vz = vel_z.clone();

        // Flops per step: N*(N-1)/2 pairs * ~20 flops/pair (3 sub + 3 mul + 2 add + 1 div + ...)
        let time_ms = bench_closure(|| {
            for _ in 0..n_steps {
                let mut ax = vec![0.0f64; n_particles];
                let mut ay = vec![0.0f64; n_particles];
                let mut az = vec![0.0f64; n_particles];

                for i in 0..n_particles {
                    for j in (i+1)..n_particles {
                        let dx = px[j] - px[i];
                        let dy = py[j] - py[i];
                        let dz = pz[j] - pz[i];
                        let dist_sq = dx*dx + dy*dy + dz*dz + epsilon;
                        let inv_dist = 1.0 / dist_sq.sqrt();
                        let f = g * mass[i] * mass[j] * inv_dist * inv_dist * inv_dist;
                        let fx = f * dx;
                        let fy = f * dy;
                        let fz = f * dz;
                        let ai = 1.0 / mass[i];
                        let aj = 1.0 / mass[j];
                        ax[i] += fx * ai; ay[i] += fy * ai; az[i] += fz * ai;
                        ax[j] -= fx * aj; ay[j] -= fy * aj; az[j] -= fz * aj;
                    }
                }
                for i in 0..n_particles {
                    vx[i] += ax[i] * dt;
                    vy[i] += ay[i] * dt;
                    vz[i] += az[i] * dt;
                    px[i] += vx[i] * dt;
                    py[i] += vy[i] * dt;
                    pz[i] += vz[i] * dt;
                }
            }
        }, 2, 10);

        time_ms
    };

    // AVX2 N-body
    #[cfg(target_arch = "x86_64")]
    let nbody_avx2_time = {
        let mut px = pos_x.clone();
        let mut py = pos_y.clone();
        let mut pz = pos_z.clone();
        let mut vx = vel_x.clone();
        let mut vy = vel_y.clone();
        let mut vz = vel_z.clone();

        let time_ms = bench_closure(|| {
            for _ in 0..n_steps {
                unsafe {
                    nbody_step_avx2(
                        &mut px, &mut py, &mut pz,
                        &mut vx, &mut vy, &mut vz,
                        &mass, g, epsilon, dt, n_particles,
                    );
                }
            }
        }, 2, 10);

        time_ms
    };

    let total_flops = (n_particles * (n_particles - 1) / 2) as f64 * 20.0 * n_steps as f64;
    let scalar_gflops = total_flops / (nbody_scalar_time * 1e-3) / 1e9;

    #[cfg(target_arch = "x86_64")]
    {
        let avx2_gflops = total_flops / (nbody_avx2_time * 1e-3) / 1e9;
        let results = vec![
            BenchResult { name: "Rust scalar".into(), time_ms: nbody_scalar_time, gflops: scalar_gflops },
            BenchResult { name: "Rust AVX2/FMA".into(), time_ms: nbody_avx2_time, gflops: avx2_gflops },
        ];
        print_comparison_table(&results, 0.0, 0.0);
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let results = vec![
            BenchResult { name: "Rust scalar".into(), time_ms: nbody_scalar_time, gflops: scalar_gflops },
        ];
        print_comparison_table(&results, 0.0, 0.0);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
unsafe fn nbody_step_avx2(
    px: &mut [f64], py: &mut [f64], pz: &mut [f64],
    vx: &mut [f64], vy: &mut [f64], vz: &mut [f64],
    mass: &[f64], g: f64, epsilon: f64, dt: f64, n: usize,
) {
    use std::arch::x86_64::*;

    let mut ax = vec![0.0f64; n];
    let mut ay = vec![0.0f64; n];
    let mut az = vec![0.0f64; n];

    for i in 0..n {
        let pix = px[i];
        let piy = py[i];
        let piz = pz[i];
        let mi = mass[i];

        // Process j in chunks of 4
        let mut j = i + 1;
        while j + 4 <= n {
            let dx = _mm256_sub_pd(_mm256_loadu_pd(px.as_ptr().add(j)), _mm256_set1_pd(pix));
            let dy = _mm256_sub_pd(_mm256_loadu_pd(py.as_ptr().add(j)), _mm256_set1_pd(piy));
            let dz = _mm256_sub_pd(_mm256_loadu_pd(pz.as_ptr().add(j)), _mm256_set1_pd(piz));

            let dist_sq = _mm256_fmadd_pd(dx, dx, _mm256_fmadd_pd(dy, dy,
                             _mm256_fmadd_pd(dz, dz, _mm256_set1_pd(epsilon))));

            // inv_dist = 1/sqrt(dist_sq) — use SIMD sqrt + reciprocal
            let inv_dist_sq = _mm256_div_pd(_mm256_set1_pd(1.0), dist_sq);
            // Newton-Raphson refine for 1/sqrt(x): not needed for bench accuracy
            let inv_dist = _mm256_sqrt_pd(inv_dist_sq);
            let inv_dist3 = _mm256_mul_pd(inv_dist_sq, inv_dist);

            let f_scale = _mm256_mul_pd(_mm256_set1_pd(g * mi), _mm256_mul_pd(
                _mm256_loadu_pd(mass.as_ptr().add(j)), inv_dist3));

            let fx = _mm256_mul_pd(f_scale, dx);
            let fy = _mm256_mul_pd(f_scale, dy);
            let fz = _mm256_mul_pd(f_scale, dz);

            // Accumulate force on i (horizontal sum)
            let ai = 1.0 / mi;
            let fx_sum = horizontal_sum_pd(fx);
            let fy_sum = horizontal_sum_pd(fy);
            let fz_sum = horizontal_sum_pd(fz);
            ax[i] += fx_sum * ai;
            ay[i] += fy_sum * ai;
            az[i] += fz_sum * ai;

            // Accumulate on j particles
            let mj = _mm256_loadu_pd(mass.as_ptr().add(j));
            let aj = _mm256_div_pd(_mm256_set1_pd(1.0), mj);
            _mm256_storeu_pd(ax.as_mut_ptr().add(j),
                _mm256_fnmadd_pd(fx, aj, _mm256_loadu_pd(ax.as_ptr().add(j))));
            _mm256_storeu_pd(ay.as_mut_ptr().add(j),
                _mm256_fnmadd_pd(fy, aj, _mm256_loadu_pd(ay.as_ptr().add(j))));
            _mm256_storeu_pd(az.as_mut_ptr().add(j),
                _mm256_fnmadd_pd(fz, aj, _mm256_loadu_pd(az.as_ptr().add(j))));

            j += 4;
        }
        // Remaining j scalar
        while j < n {
            let dx = px[j] - pix;
            let dy = py[j] - piy;
            let dz = pz[j] - piz;
            let dist_sq = dx*dx + dy*dy + dz*dz + epsilon;
            let inv_dist = 1.0 / dist_sq.sqrt();
            let f = g * mi * mass[j] * inv_dist * inv_dist * inv_dist;
            let fx = f * dx; let fy = f * dy; let fz = f * dz;
            let ai = 1.0 / mi; let aj = 1.0 / mass[j];
            ax[i] += fx * ai; ay[i] += fy * ai; az[i] += fz * ai;
            ax[j] -= fx * aj; ay[j] -= fy * aj; az[j] -= fz * aj;
            j += 1;
        }
    }

    // Update velocities and positions
    let dt_v = _mm256_set1_pd(dt);
    let mut i = 0;
    while i + 4 <= n {
        let ax4 = _mm256_loadu_pd(ax.as_ptr().add(i));
        let ay4 = _mm256_loadu_pd(ay.as_ptr().add(i));
        let az4 = _mm256_loadu_pd(az.as_ptr().add(i));
        let vx4 = _mm256_loadu_pd(vx.as_ptr().add(i));
        let vy4 = _mm256_loadu_pd(vy.as_ptr().add(i));
        let vz4 = _mm256_loadu_pd(vz.as_ptr().add(i));
        let px4 = _mm256_loadu_pd(px.as_ptr().add(i));
        let py4 = _mm256_loadu_pd(py.as_ptr().add(i));
        let pz4 = _mm256_loadu_pd(pz.as_ptr().add(i));

        let nvx = _mm256_fmadd_pd(ax4, dt_v, vx4);
        let nvy = _mm256_fmadd_pd(ay4, dt_v, vy4);
        let nvz = _mm256_fmadd_pd(az4, dt_v, vz4);
        let npx = _mm256_fmadd_pd(nvx, dt_v, px4);
        let npy = _mm256_fmadd_pd(nvy, dt_v, py4);
        let npz = _mm256_fmadd_pd(nvz, dt_v, pz4);

        _mm256_storeu_pd(vx.as_mut_ptr().add(i), nvx);
        _mm256_storeu_pd(vy.as_mut_ptr().add(i), nvy);
        _mm256_storeu_pd(vz.as_mut_ptr().add(i), nvz);
        _mm256_storeu_pd(px.as_mut_ptr().add(i), npx);
        _mm256_storeu_pd(py.as_mut_ptr().add(i), npy);
        _mm256_storeu_pd(pz.as_mut_ptr().add(i), npz);
        i += 4;
    }
    while i < n {
        vx[i] += ax[i] * dt; vy[i] += ay[i] * dt; vz[i] += az[i] * dt;
        px[i] += vx[i] * dt; py[i] += vy[i] * dt; pz[i] += vz[i] * dt;
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn horizontal_sum_pd(v: std::arch::x86_64::__m256d) -> f64 {
    use std::arch::x86_64::*;
    let hi = _mm256_extractf128_pd(v, 1);
    let lo = _mm256_castpd256_pd128(v);
    let sum = _mm_add_pd(lo, hi);
    let shuf = _mm_unpackhi_pd(sum, sum);
    let s = _mm_add_sd(sum, shuf);
    _mm_cvtsd_f64(s)
}

// ══════════════════════════════════════════════════════════════════════════════
// 3. Linear Algebra Chains
// ══════════════════════════════════════════════════════════════════════════════

fn bench_linalg_chains() {
    print_section("3. Linear Algebra Chains (128×128 matrices)");

    let dim = 128usize;
    let n = dim * dim;

    // Initialize matrices
    let make_mat = || -> Vec<f64> {
        (0..n).map(|i| (i as f64 * 0.001).sin()).collect()
    };

    // ── SympleX JIT matmul ───────────────────────────────────────────────
    println!("\n  Benchmark: C = A @ B (SympleX JIT SSE2 scalar matmul)");
    let jit_matmul_time = {
        let kernel = compile_matmul(dim, dim, dim, SimdLevel::Sse2)
            .expect("Failed to compile matmul kernel");
        let mut a = make_mat();
        let mut b = make_mat();
        let mut c = vec![0.0f64; n];

        let a_ptr = a.as_mut_ptr();
        let b_ptr = b.as_mut_ptr();
        let c_ptr = c.as_mut_ptr();
        let slot_ptrs = [c_ptr, a_ptr, b_ptr];
        let slot_sizes = [n as i64, n as i64, n as i64];

        let time_ms = bench_closure(|| {
            unsafe {
                call_jit_kernel(&kernel, &slot_ptrs, &slot_sizes);
            }
        }, 1, 5);

        let _ = kernel; // keep alive
        time_ms
    };

    // ── Scalar Rust matmul ───────────────────────────────────────────────
    let scalar_matmul_time = {
        let a = make_mat();
        let b = make_mat();
        let mut c = vec![0.0f64; n];

        let time_ms = bench_closure(|| {
            for i in 0..dim {
                for j in 0..dim {
                    let mut sum = 0.0f64;
                    for k in 0..dim {
                        sum += a[i*dim+k] * b[k*dim+j];
                    }
                    c[i*dim+j] = sum;
                }
            }
        }, 1, 5);
        time_ms
    };

    // ── AVX2 matmul (tiled) ─────────────────────────────────────────────
    #[cfg(target_arch = "x86_64")]
    let avx2_matmul_time = {
        let a = make_mat();
        let b = make_mat();
        let mut c = vec![0.0f64; n];

        let time_ms = bench_closure(|| {
            unsafe { matmul_avx2(&a, &b, &mut c, dim); }
        }, 1, 5);
        time_ms
    };

    let matmul_flops = 2.0 * (dim as f64).powi(3); // 2*M*N*K
    let jit_gflops = matmul_flops / (jit_matmul_time * 1e-3) / 1e9;
    let scalar_gflops = matmul_flops / (scalar_matmul_time * 1e-3) / 1e9;

    #[cfg(target_arch = "x86_64")]
    {
        let avx2_gflops = matmul_flops / (avx2_matmul_time * 1e-3) / 1e9;
        let results = vec![
            BenchResult { name: "SympleX JIT (SSE2)".into(), time_ms: jit_matmul_time, gflops: jit_gflops },
            BenchResult { name: "Rust scalar".into(), time_ms: scalar_matmul_time, gflops: scalar_gflops },
            BenchResult { name: "Rust AVX2/FMA".into(), time_ms: avx2_matmul_time, gflops: avx2_gflops },
        ];
        print_comparison_table(&results, 0.0, 0.0);
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let results = vec![
            BenchResult { name: "SympleX JIT (SSE2)".into(), time_ms: jit_matmul_time, gflops: jit_gflops },
            BenchResult { name: "Rust scalar".into(), time_ms: scalar_matmul_time, gflops: scalar_gflops },
        ];
        print_comparison_table(&results, 0.0, 0.0);
    }

    // ── Fused chain: result = A @ B + C @ D + E ─────────────────────────
    println!("\n  Benchmark: result = A @ B + C @ D + E (fused vs separate)");

    let fused_chain_time = {
        let a = make_mat(); let b = make_mat();
        let c = make_mat(); let d = make_mat();
        let e = make_mat();
        let mut ab = vec![0.0f64; n];
        let mut cd = vec![0.0f64; n];
        let mut result = vec![0.0f64; n];

        // Separate: 2 matmuls + 2 adds = 3 passes
        let time_ms = bench_closure(|| {
            // AB = A @ B
            for i in 0..dim {
                for j in 0..dim {
                    let mut sum = 0.0f64;
                    for k in 0..dim { sum += a[i*dim+k] * b[k*dim+j]; }
                    ab[i*dim+j] = sum;
                }
            }
            // CD = C @ D
            for i in 0..dim {
                for j in 0..dim {
                    let mut sum = 0.0f64;
                    for k in 0..dim { sum += c[i*dim+k] * d[k*dim+j]; }
                    cd[i*dim+j] = sum;
                }
            }
            // result = AB + CD + E
            for i in 0..n {
                result[i] = ab[i] + cd[i] + e[i];
            }
        }, 1, 5);
        time_ms
    };

    // Fused: compute AB and CD element-by-element, add E on the fly
    let fused_opt_time = {
        let a = make_mat(); let b = make_mat();
        let c = make_mat(); let d = make_mat();
        let e = make_mat();
        let mut result = vec![0.0f64; n];

        let time_ms = bench_closure(|| {
            // Compute AB[i,j] + CD[i,j] + E[i,j] in one pass
            // Still need two matmul accumulations but no intermediate storage
            for i in 0..dim {
                for j in 0..dim {
                    let mut ab = 0.0f64;
                    let mut cd = 0.0f64;
                    for k in 0..dim {
                        ab += a[i*dim+k] * b[k*dim+j];
                        cd += c[i*dim+k] * d[k*dim+j];
                    }
                    result[i*dim+j] = ab + cd + e[i*dim+j];
                }
            }
        }, 1, 5);
        time_ms
    };

    let chain_flops = 2.0 * 2.0 * (dim as f64).powi(3) + 2.0 * n as f64; // 2 matmuls + 2 adds
    let fused_gflops = chain_flops / (fused_chain_time * 1e-3) / 1e9;
    let fused_opt_gflops = chain_flops / (fused_opt_time * 1e-3) / 1e9;

    let results = vec![
        BenchResult { name: "Separate (3 passes)".into(), time_ms: fused_chain_time, gflops: fused_gflops },
        BenchResult { name: "Fused (1 pass)".into(), time_ms: fused_opt_time, gflops: fused_opt_gflops },
    ];
    print_comparison_table(&results, 0.0, 0.0);

    // ── Fused: (A @ B) * C + D ───────────────────────────────────────────
    println!("\n  Benchmark: result = (A @ B) * C + D (fused matmul+mul+add)");
    let fused_muladd_time = {
        let a = make_mat(); let b = make_mat();
        let c = make_mat(); let d = make_mat();
        let mut ab = vec![0.0f64; n];
        let mut result = vec![0.0f64; n];

        let time_ms = bench_closure(|| {
            for i in 0..dim {
                for j in 0..dim {
                    let mut sum = 0.0f64;
                    for k in 0..dim { sum += a[i*dim+k] * b[k*dim+j]; }
                    ab[i*dim+j] = sum;
                }
            }
            for i in 0..n { result[i] = ab[i] * c[i] + d[i]; }
        }, 1, 5);
        time_ms
    };

    let fused_muladd_opt_time = {
        let a = make_mat(); let b = make_mat();
        let c = make_mat(); let d = make_mat();
        let mut result = vec![0.0f64; n];

        let time_ms = bench_closure(|| {
            for i in 0..dim {
                for j in 0..dim {
                    let mut sum = 0.0f64;
                    for k in 0..dim { sum += a[i*dim+k] * b[k*dim+j]; }
                    result[i*dim+j] = sum * c[i*dim+j] + d[i*dim+j];
                }
            }
        }, 1, 5);
        time_ms
    };

    let muladd_flops = 2.0 * (dim as f64).powi(3) + 2.0 * n as f64;
    let sep_gflops = muladd_flops / (fused_muladd_time * 1e-3) / 1e9;
    let fused_gflops2 = muladd_flops / (fused_muladd_opt_time * 1e-3) / 1e9;
    let results = vec![
        BenchResult { name: "Separate (2 passes)".into(), time_ms: fused_muladd_time, gflops: sep_gflops },
        BenchResult { name: "Fused (1 pass)".into(), time_ms: fused_muladd_opt_time, gflops: fused_gflops2 },
    ];
    print_comparison_table(&results, 0.0, 0.0);
}

/// AVX2 tiled matmul with micro-kernel for 4×4 blocks
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
unsafe fn matmul_avx2(a: &[f64], b: &[f64], c: &mut [f64], dim: usize) {
    use std::arch::x86_64::*;

    // Zero out C
    for v in c.iter_mut() { *v = 0.0; }

    let block = 4; // micro-kernel block size
    for i in (0..dim).step_by(block) {
        for k in (0..dim).step_by(block) {
            for j in (0..dim).step_by(block) {
                // Micro-kernel: accumulate block
                let i_end = (i + block).min(dim);
                let k_end = (k + block).min(dim);
                let j_end = (j + block).min(dim);

                for ii in i..i_end {
                    for jj in (j..j_end).step_by(4) {
                        if jj + 4 <= j_end {
                            let mut acc = _mm256_loadu_pd(c.as_ptr().add(ii * dim + jj));
                            for kk in k..k_end {
                                let a_val = _mm256_set1_pd(a[ii * dim + kk]);
                                let b_row = _mm256_loadu_pd(b.as_ptr().add(kk * dim + jj));
                                acc = _mm256_fmadd_pd(a_val, b_row, acc);
                            }
                            _mm256_storeu_pd(c.as_mut_ptr().add(ii * dim + jj), acc);
                        } else {
                            for jjj in jj..j_end {
                                let mut acc = c[ii * dim + jjj];
                                for kk in k..k_end {
                                    acc += a[ii * dim + kk] * b[kk * dim + jjj];
                                }
                                c[ii * dim + jjj] = acc;
                            }
                        }
                    }
                }
            }
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// 4. Physics Integrators (Euler, RK4)
// ══════════════════════════════════════════════════════════════════════════════

fn bench_integrators() {
    print_section("4. Physics Integrators — Euler & RK4 (Lorenz, 100000 steps)");

    // Lorenz system: dx/dt = sigma*(y - x), dy/dt = x*(rho - z) - y, dz/dt = x*y - beta*z
    let sigma = 10.0f64;
    let rho = 28.0f64;
    let beta = 8.0 / 3.0;
    let dt = 0.01f64;
    let n_steps = 100000usize;

    let lorenz = |x: f64, y: f64, z: f64| -> (f64, f64, f64) {
        (sigma * (y - x), x * (rho - z) - y, x * y - beta * z)
    };

    // ── Euler ────────────────────────────────────────────────────────────
    let euler_time = {
        let mut x = 1.0f64; let mut y = 1.0f64; let mut z = 1.0f64;

        let time_ms = bench_closure(|| {
            for _ in 0..n_steps {
                let (dx, dy, dz) = lorenz(x, y, z);
                x += dt * dx;
                y += dt * dy;
                z += dt * dz;
            }
            std::hint::black_box((x, y, z));
            // Reset for next iteration
            x = 1.0; y = 1.0; z = 1.0;
        }, 3, 15);
        time_ms
    };

    // ── RK4 ──────────────────────────────────────────────────────────────
    let rk4_time = {
        let mut x = 1.0f64; let mut y = 1.0f64; let mut z = 1.0f64;

        let time_ms = bench_closure(|| {
            for _ in 0..n_steps {
                let (k1x, k1y, k1z) = lorenz(x, y, z);
                let (k2x, k2y, k2z) = lorenz(
                    x + 0.5*dt*k1x, y + 0.5*dt*k1y, z + 0.5*dt*k1z);
                let (k3x, k3y, k3z) = lorenz(
                    x + 0.5*dt*k2x, y + 0.5*dt*k2y, z + 0.5*dt*k2z);
                let (k4x, k4y, k4z) = lorenz(
                    x + dt*k3x, y + dt*k3y, z + dt*k3z);
                x += dt/6.0 * (k1x + 2.0*k2x + 2.0*k3x + k4x);
                y += dt/6.0 * (k1y + 2.0*k2y + 2.0*k3y + k4y);
                z += dt/6.0 * (k1z + 2.0*k2z + 2.0*k3z + k4z);
            }
            std::hint::black_box((x, y, z));
            x = 1.0; y = 1.0; z = 1.0;
        }, 3, 15);
        time_ms
    };

    // Euler: ~9 flops/step, RK4: ~4*9 + 6*3 = 54 flops/step
    let euler_flops = 9.0 * n_steps as f64;
    let rk4_flops = 54.0 * n_steps as f64;

    println!("\n  Lorenz System (3D chaotic ODE): σ=10, ρ=28, β=8/3");
    let results = vec![
        BenchResult { name: "Euler (scalar)".into(), time_ms: euler_time, gflops: euler_flops / (euler_time * 1e-3) / 1e9 },
        BenchResult { name: "RK4 (scalar)".into(), time_ms: rk4_time, gflops: rk4_flops / (rk4_time * 1e-3) / 1e9 },
    ];
    print_comparison_table(&results, 0.0, 0.0);

    // ── 2D Diffusion (explicit Euler on grid) ────────────────────────────
    println!("\n  2D Diffusion: Explicit Euler on 256×256 grid, 100 steps");
    let nx = 256; let ny = 256;
    let alpha_d = 0.01f64;
    let dt_d = 0.1f64;
    let coeff_d = alpha_d * dt_d;

    let mut u_diff = vec![0.0f64; nx * ny];
    for i in 0..nx {
        for j in 0..ny {
            let dx = i as f64 - nx as f64 / 2.0;
            let dy = j as f64 - ny as f64 / 2.0;
            u_diff[i * ny + j] = (-0.5 * (dx * dx + dy * dy) / 100.0).exp();
        }
    }

    let diff_scalar_time = {
        let mut u = u_diff.clone();
        let mut u_new = u_diff.clone();
        let steps = 100;

        let time_ms = bench_closure(|| {
            for _ in 0..steps {
                for i in 1..nx-1 {
                    for j in 1..ny-1 {
                        let idx = i * ny + j;
                        u_new[idx] = u[idx] + coeff_d * (
                            u[(i-1)*ny+j] + u[(i+1)*ny+j] +
                            u[i*ny+(j-1)] + u[i*ny+(j+1)] - 4.0*u[idx]);
                    }
                }
                std::mem::swap(&mut u, &mut u_new);
            }
        }, 2, 10);
        time_ms
    };

    #[cfg(target_arch = "x86_64")]
    let diff_avx2_time = {
        let mut u = u_diff.clone();
        let mut u_new = u_diff.clone();
        let steps = 100;

        let time_ms = bench_closure(|| {
            for _ in 0..steps {
                unsafe { heat_stencil_avx2(&u, &mut u_new, nx, ny, coeff_d); }
                std::mem::swap(&mut u, &mut u_new);
            }
        }, 2, 10);
        time_ms
    };

    let diff_flops = 10.0 * (nx - 2) as f64 * (ny - 2) as f64 * 100.0;
    let diff_scalar_gflops = diff_flops / (diff_scalar_time * 1e-3) / 1e9;

    #[cfg(target_arch = "x86_64")]
    {
        let diff_avx2_gflops = diff_flops / (diff_avx2_time * 1e-3) / 1e9;
        let results = vec![
            BenchResult { name: "Euler scalar".into(), time_ms: diff_scalar_time, gflops: diff_scalar_gflops },
            BenchResult { name: "Euler AVX2/FMA".into(), time_ms: diff_avx2_time, gflops: diff_avx2_gflops },
        ];
        print_comparison_table(&results, 0.0, 0.0);
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let results = vec![
            BenchResult { name: "Euler scalar".into(), time_ms: diff_scalar_time, gflops: diff_scalar_gflops },
        ];
        print_comparison_table(&results, 0.0, 0.0);
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// 5. Tensor Field Operations
// ══════════════════════════════════════════════════════════════════════════════

fn bench_tensor_fields() {
    print_section("5. Tensor Field Operations (512×512 grid)");

    let nx = 512usize;
    let ny = 512usize;
    let n = nx * ny;
    let dx = 0.01f64;

    // Initialize scalar field f(x,y) = sin(x) * cos(y)
    let mut f = vec![0.0f64; n];
    let mut u_vec = vec![0.0f64; n];
    let mut v_vec = vec![0.0f64; n];
    for i in 0..nx {
        for j in 0..ny {
            let x = i as f64 * dx;
            let y = j as f64 * dx;
            f[i * ny + j] = x.sin() * y.cos();
            u_vec[i * ny + j] = (x * y).cos();
            v_vec[i * ny + j] = (x + y).sin();
        }
    }

    // ── Gradient ─────────────────────────────────────────────────────────
    let grad_scalar_time = {
        let mut grad_x = vec![0.0f64; n];
        let mut grad_y = vec![0.0f64; n];
        let inv_2dx = 1.0 / (2.0 * dx);

        let time_ms = bench_closure(|| {
            for i in 1..nx-1 {
                for j in 1..ny-1 {
                    grad_x[i*ny+j] = (f[(i+1)*ny+j] - f[(i-1)*ny+j]) * inv_2dx;
                    grad_y[i*ny+j] = (f[i*ny+(j+1)] - f[i*ny+(j-1)]) * inv_2dx;
                }
            }
        }, 3, 15);
        time_ms
    };

    #[cfg(target_arch = "x86_64")]
    let grad_avx2_time = {
        let mut grad_x = vec![0.0f64; n];
        let mut grad_y = vec![0.0f64; n];

        let time_ms = bench_closure(|| {
            unsafe { gradient_avx2(&f, &mut grad_x, &mut grad_y, nx, ny, dx); }
        }, 3, 15);
        time_ms
    };

    // ── Divergence ───────────────────────────────────────────────────────
    let div_scalar_time = {
        let mut div = vec![0.0f64; n];
        let inv_2dx = 1.0 / (2.0 * dx);

        let time_ms = bench_closure(|| {
            for i in 1..nx-1 {
                for j in 1..ny-1 {
                    let du_dx = (u_vec[(i+1)*ny+j] - u_vec[(i-1)*ny+j]) * inv_2dx;
                    let dv_dy = (v_vec[i*ny+(j+1)] - v_vec[i*ny+(j-1)]) * inv_2dx;
                    div[i*ny+j] = du_dx + dv_dy;
                }
            }
        }, 3, 15);
        time_ms
    };

    #[cfg(target_arch = "x86_64")]
    let div_avx2_time = {
        let mut div = vec![0.0f64; n];
        let time_ms = bench_closure(|| {
            unsafe { divergence_avx2(&u_vec, &v_vec, &mut div, nx, ny, dx); }
        }, 3, 15);
        time_ms
    };

    // ── Curl (2D scalar) ─────────────────────────────────────────────────
    let curl_scalar_time = {
        let mut curl = vec![0.0f64; n];
        let inv_2dx = 1.0 / (2.0 * dx);

        let time_ms = bench_closure(|| {
            for i in 1..nx-1 {
                for j in 1..ny-1 {
                    let dv_dx = (v_vec[(i+1)*ny+j] - v_vec[(i-1)*ny+j]) * inv_2dx;
                    let du_dy = (u_vec[i*ny+(j+1)] - u_vec[i*ny+(j-1)]) * inv_2dx;
                    curl[i*ny+j] = dv_dx - du_dy;
                }
            }
        }, 3, 15);
        time_ms
    };

    #[cfg(target_arch = "x86_64")]
    let curl_avx2_time = {
        let mut curl = vec![0.0f64; n];
        let time_ms = bench_closure(|| {
            unsafe { curl_avx2(&u_vec, &v_vec, &mut curl, nx, ny, dx); }
        }, 3, 15);
        time_ms
    };

    // ── Laplacian ────────────────────────────────────────────────────────
    let lap_scalar_time = {
        let mut lap = vec![0.0f64; n];
        let inv_dx2 = 1.0 / (dx * dx);

        let time_ms = bench_closure(|| {
            for i in 1..nx-1 {
                for j in 1..ny-1 {
                    lap[i*ny+j] = (f[(i+1)*ny+j] + f[(i-1)*ny+j] +
                                   f[i*ny+(j+1)] + f[i*ny+(j-1)] - 4.0*f[i*ny+j]) * inv_dx2;
                }
            }
        }, 3, 15);
        time_ms
    };

    #[cfg(target_arch = "x86_64")]
    let lap_avx2_time = {
        let mut lap = vec![0.0f64; n];
        let time_ms = bench_closure(|| {
            unsafe { laplacian_avx2(&f, &mut lap, nx, ny, dx); }
        }, 3, 15);
        time_ms
    };

    // Report
    let interior = (nx - 2) as f64 * (ny - 2) as f64;

    println!("\n  grad_x = (f[i+1,j] - f[i-1,j]) / (2·dx), same for y");
    println!("  div = du/dx + dv/dy");
    println!("  curl = dv/dx - du/dy");
    println!("  lap = (f[i+1,j] + f[i-1,j] + f[i,j+1] + f[i,j-1] - 4·f[i,j]) / dx²");

    // Gradient: 2 subtractions + 2 mults per point = 4 flops
    let grad_flops = 4.0 * interior;
    let grad_scalar_gflops = grad_flops / (grad_scalar_time * 1e-3) / 1e9;
    #[cfg(target_arch = "x86_64")]
    let grad_avx2_gflops = grad_flops / (grad_avx2_time * 1e-3) / 1e9;

    // Divergence: 4 subtractions + 2 mults + 1 add = 7 flops
    let div_flops = 7.0 * interior;
    let div_scalar_gflops = div_flops / (div_scalar_time * 1e-3) / 1e9;
    #[cfg(target_arch = "x86_64")]
    let div_avx2_gflops = div_flops / (div_avx2_time * 1e-3) / 1e9;

    // Curl: same as div
    let curl_flops = 7.0 * interior;
    let curl_scalar_gflops = curl_flops / (curl_scalar_time * 1e-3) / 1e9;
    #[cfg(target_arch = "x86_64")]
    let curl_avx2_gflops = curl_flops / (curl_avx2_time * 1e-3) / 1e9;

    // Laplacian: 4 adds + 1 sub + 1 mult = 6 flops
    let lap_flops = 6.0 * interior;
    let lap_scalar_gflops = lap_flops / (lap_scalar_time * 1e-3) / 1e9;
    #[cfg(target_arch = "x86_64")]
    let lap_avx2_gflops = lap_flops / (lap_avx2_time * 1e-3) / 1e9;

    #[cfg(target_arch = "x86_64")]
    {
        let results = vec![
            BenchResult { name: "Gradient scalar".into(), time_ms: grad_scalar_time, gflops: grad_scalar_gflops },
            BenchResult { name: "Gradient AVX2".into(), time_ms: grad_avx2_time, gflops: grad_avx2_gflops },
            BenchResult { name: "Divergence scalar".into(), time_ms: div_scalar_time, gflops: div_scalar_gflops },
            BenchResult { name: "Divergence AVX2".into(), time_ms: div_avx2_time, gflops: div_avx2_gflops },
            BenchResult { name: "Curl scalar".into(), time_ms: curl_scalar_time, gflops: curl_scalar_gflops },
            BenchResult { name: "Curl AVX2".into(), time_ms: curl_avx2_time, gflops: curl_avx2_gflops },
            BenchResult { name: "Laplacian scalar".into(), time_ms: lap_scalar_time, gflops: lap_scalar_gflops },
            BenchResult { name: "Laplacian AVX2".into(), time_ms: lap_avx2_time, gflops: lap_avx2_gflops },
        ];
        print_comparison_table(&results, 0.0, 0.0);
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let results = vec![
            BenchResult { name: "Gradient scalar".into(), time_ms: grad_scalar_time, gflops: grad_scalar_gflops },
            BenchResult { name: "Divergence scalar".into(), time_ms: div_scalar_time, gflops: div_scalar_gflops },
            BenchResult { name: "Curl scalar".into(), time_ms: curl_scalar_time, gflops: curl_scalar_gflops },
            BenchResult { name: "Laplacian scalar".into(), time_ms: lap_scalar_time, gflops: lap_scalar_gflops },
        ];
        print_comparison_table(&results, 0.0, 0.0);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
unsafe fn gradient_avx2(f: &[f64], grad_x: &mut [f64], grad_y: &mut [f64], nx: usize, ny: usize, dx: f64) {
    use std::arch::x86_64::*;
    let inv_2dx = _mm256_set1_pd(0.5 / dx);

    for i in 1..nx-1 {
        let row_c = i * ny;
        let row_up = (i - 1) * ny;
        let row_dn = (i + 1) * ny;

        let mut j = 1;
        while j + 4 <= ny - 1 {
            let f_plus_i  = _mm256_loadu_pd(f.as_ptr().add(row_dn + j));
            let f_minus_i = _mm256_loadu_pd(f.as_ptr().add(row_up + j));
            let f_plus_j  = _mm256_loadu_pd(f.as_ptr().add(row_c + j + 1));
            let f_minus_j = _mm256_loadu_pd(f.as_ptr().add(row_c + j - 1));

            let gx = _mm256_mul_pd(_mm256_sub_pd(f_plus_i, f_minus_i), inv_2dx);
            let gy = _mm256_mul_pd(_mm256_sub_pd(f_plus_j, f_minus_j), inv_2dx);

            _mm256_storeu_pd(grad_x.as_mut_ptr().add(row_c + j), gx);
            _mm256_storeu_pd(grad_y.as_mut_ptr().add(row_c + j), gy);
            j += 4;
        }
        while j < ny - 1 {
            let idx = row_c + j;
            grad_x[idx] = (f[row_dn+j] - f[row_up+j]) * 0.5 / dx;
            grad_y[idx] = (f[idx+1] - f[idx-1]) * 0.5 / dx;
            j += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
unsafe fn divergence_avx2(u: &[f64], v: &[f64], div: &mut [f64], nx: usize, ny: usize, dx: f64) {
    use std::arch::x86_64::*;
    let inv_2dx = _mm256_set1_pd(0.5 / dx);

    for i in 1..nx-1 {
        let row_c = i * ny;
        let row_up = (i - 1) * ny;
        let row_dn = (i + 1) * ny;

        let mut j = 1;
        while j + 4 <= ny - 1 {
            let du_dx = _mm256_mul_pd(_mm256_sub_pd(
                _mm256_loadu_pd(u.as_ptr().add(row_dn + j)),
                _mm256_loadu_pd(u.as_ptr().add(row_up + j))), inv_2dx);
            let dv_dy = _mm256_mul_pd(_mm256_sub_pd(
                _mm256_loadu_pd(v.as_ptr().add(row_c + j + 1)),
                _mm256_loadu_pd(v.as_ptr().add(row_c + j - 1))), inv_2dx);

            _mm256_storeu_pd(div.as_mut_ptr().add(row_c + j),
                _mm256_add_pd(du_dx, dv_dy));
            j += 4;
        }
        while j < ny - 1 {
            let idx = row_c + j;
            let du_dx = (u[row_dn+j] - u[row_up+j]) * 0.5 / dx;
            let dv_dy = (v[idx+1] - v[idx-1]) * 0.5 / dx;
            div[idx] = du_dx + dv_dy;
            j += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
unsafe fn curl_avx2(u: &[f64], v: &[f64], curl: &mut [f64], nx: usize, ny: usize, dx: f64) {
    use std::arch::x86_64::*;
    let inv_2dx = _mm256_set1_pd(0.5 / dx);

    for i in 1..nx-1 {
        let row_c = i * ny;
        let row_up = (i - 1) * ny;
        let row_dn = (i + 1) * ny;

        let mut j = 1;
        while j + 4 <= ny - 1 {
            let dv_dx = _mm256_mul_pd(_mm256_sub_pd(
                _mm256_loadu_pd(v.as_ptr().add(row_dn + j)),
                _mm256_loadu_pd(v.as_ptr().add(row_up + j))), inv_2dx);
            let du_dy = _mm256_mul_pd(_mm256_sub_pd(
                _mm256_loadu_pd(u.as_ptr().add(row_c + j + 1)),
                _mm256_loadu_pd(u.as_ptr().add(row_c + j - 1))), inv_2dx);

            _mm256_storeu_pd(curl.as_mut_ptr().add(row_c + j),
                _mm256_sub_pd(dv_dx, du_dy));
            j += 4;
        }
        while j < ny - 1 {
            let idx = row_c + j;
            let dv_dx = (v[row_dn+j] - v[row_up+j]) * 0.5 / dx;
            let du_dy = (u[idx+1] - u[idx-1]) * 0.5 / dx;
            curl[idx] = dv_dx - du_dy;
            j += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[target_feature(enable = "fma")]
unsafe fn laplacian_avx2(f: &[f64], lap: &mut [f64], nx: usize, ny: usize, dx: f64) {
    use std::arch::x86_64::*;
    let inv_dx2 = _mm256_set1_pd(1.0 / (dx * dx));
    let neg4 = _mm256_set1_pd(-4.0);

    for i in 1..nx-1 {
        let row_c = i * ny;
        let row_up = (i - 1) * ny;
        let row_dn = (i + 1) * ny;

        let mut j = 1;
        while j + 4 <= ny - 1 {
            let center = _mm256_loadu_pd(f.as_ptr().add(row_c + j));
            let up     = _mm256_loadu_pd(f.as_ptr().add(row_up + j));
            let down   = _mm256_loadu_pd(f.as_ptr().add(row_dn + j));
            let left   = _mm256_loadu_pd(f.as_ptr().add(row_c + j - 1));
            let right  = _mm256_loadu_pd(f.as_ptr().add(row_c + j + 1));

            let stencil = _mm256_fmadd_pd(neg4, center, _mm256_add_pd(
                _mm256_add_pd(up, down),
                _mm256_add_pd(left, right),
            ));
            _mm256_storeu_pd(lap.as_mut_ptr().add(row_c + j),
                _mm256_mul_pd(stencil, inv_dx2));
            j += 4;
        }
        while j < ny - 1 {
            let idx = row_c + j;
            lap[idx] = (f[row_up+j] + f[row_dn+j] + f[idx-1] + f[idx+1] - 4.0*f[idx]) / (dx*dx);
            j += 1;
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Main entry point
// ══════════════════════════════════════════════════════════════════════════════

fn main() {
    println!();
    println!("╔═══════════════════════════════════════════════════════════════════╗");
    println!("║         SympleX Physics Benchmark Suite                          ║");
    println!("║         JIT (SSE2 scalar) vs Rust AVX2/FMA vs NumPy             ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝");

    // Detect SIMD level
    let simd = detect_simd_level();
    println!();
    println!("  Detected SIMD level: {:?}", simd);
    println!("  SympleX JIT uses: SSE2 scalar (emitted x86-64 machine code)");
    println!("  Rust native uses:  AVX2 + FMA intrinsics (std::arch::x86_64)");
    println!("  NumPy reference:   Run bench_physics_numpy.py separately");
    println!();

    warmup();

    bench_pde_stencils();
    bench_nbody();
    bench_linalg_chains();
    bench_integrators();
    bench_tensor_fields();

    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  Benchmark complete. For NumPy comparison numbers, run:");
    println!("    python3 benchmarks/bench_physics_numpy.py");
    println!("═══════════════════════════════════════════════════════════════════");
}
