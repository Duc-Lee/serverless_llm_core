import os
import subprocess
import sys
import platform

def run_cmd(cmd, cwd=None):
    print(f"[*] Running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    if result.returncode != 0:
        print(f"[!] Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    build_dir = os.path.join(root_dir, "build")
    
    print("==================================================")
    print("  HydraServerless Build & Benchmark Test Script   ")
    print("==================================================")
    
    # 1. Build C++ Backend
    print("\n--- Step 1: Compiling C++ Engine via CMake ---")
    run_cmd(f"cmake -B {build_dir} -DCMAKE_BUILD_TYPE=Release", cwd=root_dir)
    run_cmd(f"cmake --build {build_dir} --config Release", cwd=root_dir)
    
    # 2. Run Benchmarks
    print("\n--- Step 2: Running End-to-End Pipelined Benchmark ---")
    
    # Detect OS to select correct binary path
    if platform.system() == "Windows":
        bench_exe = os.path.join(build_dir, "Release", "bench_end_to_end.exe")
        peeker_exe = os.path.join(build_dir, "Release", "peeker_daemon.exe")
    else:
        bench_exe = os.path.join(build_dir, "bench_end_to_end")
        peeker_exe = os.path.join(build_dir, "peeker_daemon")
        
    if os.path.exists(bench_exe):
        # Create a dummy weights file so the GDS loader has something to open
        dummy_weights = os.path.join(root_dir, "dummy_weights.bin")
        if not os.path.exists(dummy_weights):
            print("[*] Creating 1MB dummy_weights.bin for I/O simulation...")
            with open(dummy_weights, "wb") as f:
                f.write(b"0" * (1024 * 1024))
                
        run_cmd(bench_exe, cwd=root_dir)
    else:
        print(f"[!] Benchmark executable not found at {bench_exe}")
        print("    (Did the build fail due to missing CUDA Toolkit?)")
        
    # 3. Run Peeker Daemon Mock
    print("\n--- Step 3: Running eBPF Peeker Daemon Mock ---")
    if os.path.exists(peeker_exe):
        run_cmd(peeker_exe, cwd=root_dir)
    else:
        print(f"[!] Peeker executable not found at {peeker_exe}")

if __name__ == "__main__":
    main()
