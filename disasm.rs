// Disassemble the generated matmul kernel to debug EVEX encoding
use std::process::Command;

mod matmul_kernel;
use matmul_kernel::MatmulKernel;

fn main() {
    let kernel = MatmulKernel::compile(16, 16, 16);
    let code = kernel.code_bytes();
    
    println!("Generated {} bytes of machine code:", code.len());
    for (i, chunk) in code.chunks(16).enumerate() {
        print!("  {:04x}: ", i * 16);
        for &b in chunk {
            print!("{:02x} ", b);
        }
        println!();
    }
    
    // Also write to a file for objdump
    let out_path = "/tmp/symplex_matmul.bin";
    std::fs::write(out_path, code).expect("Failed to write binary");
    println!("\nWrote binary to {} for objdump disassembly", out_path);
    
    // Try to disassemble with objdump
    let output = Command::new("objdump")
        .args(["-D", "-b", "binary", "-m", "i386:x86-64", "-M", "intel", out_path])
        .output();
    
    match output {
        Ok(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            println!("\nDisassembly:\n{}", stdout);
        }
        Err(e) => {
            println!("objdump not available: {}", e);
        }
    }
}
