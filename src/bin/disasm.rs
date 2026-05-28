use matmul_kernel::MatmulKernel;

fn main() {
    let kernel = MatmulKernel::compile(4, 4, 4);
    let code = kernel.code_bytes();
    
    println!("Generated {} bytes of machine code:", code.len());
    for (i, chunk) in code.chunks(16).enumerate() {
        print!("  {:04x}: ", i * 16);
        for &b in chunk {
            print!("{:02x} ", b);
        }
        println!();
    }
    
    std::fs::write("/tmp/symplex_matmul.bin", code).expect("write failed");
    println!("\nWrote binary to /tmp/symplex_matmul.bin");
}
