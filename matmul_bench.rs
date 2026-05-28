// SympleX Matmul Kernel Benchmark
//
// Standalone benchmark that compiles and runs the AVX-512 FMA matmul JIT kernel.
// Compares against a reference scalar implementation and reports correctness + performance.
//
// Build:  rustc -O2 -C target-cpu=native matmul_bench.rs -o matmul_bench
// Run:    ./matmul_bench

use std::time::Instant;

mod matmul_kernel;

use matmul_kernel::MatmulKernel;

/// Reference scalar matmul for correctness checking
fn scalar_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for p in 0..k {
            let a_val = a[i * k + p];
            for j in 0..n {
                c[i * n + j] += a_val * b[p * n + j];
            }
        }
    }
    c
}

fn benchmark(size: usize) {
    let m = size;
    let n = size;
    let k = size;

    println!("\n═══════════════════════════════════════════════════════════");
    println!("  SympleX Matmul Benchmark: {}×{}×{} (f32)", m, n, k);
    println!("═══════════════════════════════════════════════════════════");

    // Generate random-ish data
    let a: Vec<f32> = (0..m*k).map(|i| ((i * 7 + 13) % 97) as f32 / 10.0).collect();
    let b: Vec<f32> = (0..k*n).map(|i| ((i * 11 + 17) % 89) as f32 / 10.0).collect();

    // ── Scalar reference ──
    let t0 = Instant::now();
    let c_ref = scalar_matmul(&a, &b, m, n, k);
    let scalar_us = t0.elapsed().as_micros();
    println!("  Scalar reference:  {} µs", scalar_us);

    // ── Compile JIT kernel ──
    let t0 = Instant::now();
    let mut kernel = MatmulKernel::compile(m, n, k);
    let compile_us = t0.elapsed().as_micros();
    println!("  JIT compile:       {} µs  ({} bytes)", compile_us, kernel.code_size());

    // ── Link kernel (allocate executable memory) ──
    match kernel.link() {
        Ok(()) => {
            println!("  JIT link:          OK");

            // ── Execute JIT kernel (multi-iteration for stable timing) ──
            let iters = if size <= 32 { 1000 } else if size <= 128 { 100 } else if size <= 512 { 10 } else { 3 };
            let mut c_jit = vec![0.0f32; m * n];
            // Warmup
            let _ = unsafe {
                kernel.execute(a.as_ptr(), b.as_ptr(), c_jit.as_mut_ptr(), m, n, k)
            };
            let t0 = Instant::now();
            for _ in 0..iters {
                let result = unsafe {
                    kernel.execute(a.as_ptr(), b.as_ptr(), c_jit.as_mut_ptr(), m, n, k)
                };
                std::hint::black_box(&result);
            }
            let total_us = t0.elapsed().as_micros();
            let jit_us = total_us / iters as u128;
            println!("  JIT execute:       {} µs  ({} iters)", jit_us, iters);

            // ── Check correctness ──
            let mut max_err = 0.0f32;
            let mut err_count = 0;
            for i in 0..m * n {
                let err = (c_jit[i] - c_ref[i]).abs();
                if err > max_err { max_err = err; }
                if err > 0.01 { err_count += 1; }
            }
            if err_count == 0 {
                println!("  ✓ Correctness: PASS (max error = {:.6})", max_err);
            } else {
                println!("  ✗ Correctness: FAIL ({} errors, max = {:.6})", err_count, max_err);
                // Show first few errors
                let mut shown = 0;
                for i in 0..m*n {
                    let err = (c_jit[i] - c_ref[i]).abs();
                    if err > 0.01 && shown < 5 {
                        println!("    C[{}]: JIT={} REF={} err={}", i, c_jit[i], c_ref[i], err);
                        shown += 1;
                    }
                }
            }

            // ── Speedup ──
            if jit_us > 0 && scalar_us > 0 {
                let speedup = scalar_us as f64 / jit_us as f64;
                println!("  Speedup vs scalar: {:.2}×", speedup);
            }

            // ── GFLOPS ──
            let flops = 2.0 * m as f64 * n as f64 * k as f64;
            if jit_us > 0 {
                let gflops = flops / (jit_us as f64 * 1e-6) / 1e9;
                println!("  GFLOPS (JIT):      {:.2}", gflops);
            }
            let gflops_scalar = flops / (scalar_us as f64 * 1e-6) / 1e9;
            println!("  GFLOPS (scalar):   {:.2}", gflops_scalar);
        }
        Err(e) => {
            println!("  JIT link FAILED: {}", e);
            println!("  (This is expected on systems without AVX-512 support)");
        }
    }
}

fn main() {
    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║  SympleX AVX-512 FMA Matmul JIT — Standalone Benchmark  ║");
    println!("╚═══════════════════════════════════════════════════════════╝");

    // CPU feature detection
    #[cfg(target_arch = "x86_64")]
    {
        println!("\nCPU Features:");
        println!("  AVX-512F: {}", is_x86_feature_detected!("avx512f"));
        println!("  AVX2:     {}", is_x86_feature_detected!("avx2"));
        println!("  FMA:      {}", is_x86_feature_detected!("fma"));
    }

    // Run benchmarks at different sizes
    benchmark(4);     // Tiny (for correctness)
    benchmark(16);    // Small (one ZMM block)
    benchmark(32);    // Medium
    benchmark(64);    // Large
    benchmark(128);   // Very large (cache pressure)
    benchmark(256);   // L2 cache
    benchmark(512);   // L3 cache
    benchmark(1024);  // Beyond L3

    println!("\n═══════════════════════════════════════════════════════════");
    println!("  Benchmark complete!");
    println!("═══════════════════════════════════════════════════════════");
}
