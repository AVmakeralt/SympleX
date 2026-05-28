// SympleX Poly Engine — Rust Integration Tests

#[cfg(test)]
mod tests {
    use symplex_engine::types::{Instr, BinOpKind, serialize_instr, deserialize_instr};
    use symplex_engine::polyhedral::{
        HardwareTarget, configure_extreme_ml_kernel, GuardTable,
    };

    #[test]
    fn test_micro_kernel_config() {
        let hw = HardwareTarget::ServerX86;
        let config = configure_extreme_ml_kernel(&hw, 4);
        assert_eq!(config.tile_m, 8);
        assert_eq!(config.tile_n, 8);
        assert!(config.accumulator_registers > 0);
    }

    #[test]
    fn test_serialize_deserialize_roundtrip() {
        let instrs = vec![
            Instr::LoadI64(0, 42),
            Instr::LoadI64(1, 100),
            Instr::BinOp(2, BinOpKind::Add, 0, 1),
            Instr::Move(3, 2),
            Instr::Nop,
        ];

        for instr in &instrs {
            let bytes = serialize_instr(instr);
            let (decoded, consumed) = deserialize_instr(&bytes).unwrap();
            assert_eq!(*instr, decoded);
            assert_eq!(consumed, bytes.len());
        }
    }

    #[test]
    fn test_guard_table() {
        let mut gt = GuardTable::new();
        gt.add_guard(5, 0, 100);
        gt.add_guard(10, 1, 200);
        assert_eq!(gt.guards.len(), 2);
    }

    #[test]
    fn test_cuda_ptx_generation_elementwise() {
        use symplex_engine::cuda_backend::PtxGenerator;
        let gen = PtxGenerator::default_arch();
        let kernel = gen.gen_elementwise(BinOpKind::Add, 1024);
        assert!(kernel.ptx_source.contains("symplex_elementwise"));
        assert!(kernel.ptx_source.contains("add.f32"));
        assert!(kernel.ptx_size > 0);
    }

    #[test]
    fn test_cuda_ptx_generation_matmul() {
        use symplex_engine::cuda_backend::PtxGenerator;
        let gen = PtxGenerator::default_arch();
        let kernel = gen.gen_matmul(128, 128, 64, 32, 32, 8);
        assert!(kernel.ptx_source.contains("symplex_matmul"));
        assert!(kernel.ptx_source.contains("smem_A"));
        assert!(kernel.ptx_source.contains("smem_B"));
        assert!(kernel.estimated_gflops > 0.0);
    }

    #[test]
    fn test_cuda_ptx_generation_fma() {
        use symplex_engine::cuda_backend::PtxGenerator;
        let gen = PtxGenerator::default_arch();
        let kernel = gen.gen_fma(512);
        assert!(kernel.ptx_source.contains("symplex_fma"));
        assert!(kernel.ptx_source.contains("mad.f32"));
    }

    #[test]
    fn test_cuda_ptx_generation_reduction() {
        use symplex_engine::cuda_backend::PtxGenerator;
        let gen = PtxGenerator::default_arch();
        let kernel = gen.gen_reduction(BinOpKind::Add, 1024);
        assert!(kernel.ptx_source.contains("symplex_reduction"));
        assert!(kernel.ptx_source.contains("add.f32"));
    }

    #[test]
    fn test_cuda_runtime_not_available_without_feature() {
        use symplex_engine::cuda_backend::CudaRuntime;
        if !cfg!(feature = "cuda") {
            assert!(!CudaRuntime::is_available());
        }
    }

    #[test]
    fn test_cuda_launch_config() {
        use symplex_engine::cuda_backend::LaunchConfig;
        let config = LaunchConfig::elementwise(1024, 256);
        assert_eq!(config.grid.0, 4);
        assert_eq!(config.block.0, 256);
    }

    #[test]
    fn test_cuda_gpu_device_info() {
        use symplex_engine::cuda_backend::GpuDeviceInfo;
        let info = GpuDeviceInfo {
            name: "NVIDIA A100".to_string(),
            compute_capability_major: 8,
            compute_capability_minor: 0,
            total_memory_bytes: 80 * 1024 * 1024 * 1024,
            num_sms: 108,
            warp_size: 32,
            max_threads_per_block: 1024,
            max_shared_memory_per_block: 49152,
            clock_mhz: 1410,
            memory_bandwidth_mbs: 2039000,
        };
        assert_eq!(info.sm_arch(), "sm_80");
        assert_eq!(info.num_cuda_cores(), 108 * 64); // SM 8.0 = 64 cores/SM
    }
}
