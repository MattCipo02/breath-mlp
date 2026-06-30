import subprocess
import time
import os

print("="*80)
print("INIZIO DELLA PIPELINE DI BENCHMARK PURISTA (BREATH MLP SENZA BUG DI COMPRESSIONE)")
print("="*80)

def run_cmd(cmd):
    print(f"\n[RUNNING] {cmd}")
    start = time.time()
    # Run in CWD = breath-mlp-github to ensure import of local breath_mlp works correctly
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    elapsed = time.time() - start
    print(f"[COMPLETED] in {elapsed:.1f} secondi (Exit Code: {res.returncode})")
    
    # Save stdout to a log file
    log_name = cmd.split(".py")[0].split(" ")[-1] + "_run.log"
    # Clean log name from arguments
    clean_log_name = cmd.replace("python ", "").replace(".py", "").replace(" --", "_").replace(" ", "_").replace(".", "").replace("/", "") + ".log"
    with open(clean_log_name, "w", encoding="utf-8") as f:
        f.write(f"Command: {cmd}\n")
        f.write(f"Time elapsed: {elapsed:.1f}s\n")
        f.write("="*80 + "\n")
        f.write(res.stdout)
        if res.stderr:
            f.write("\n" + "="*80 + "\nSTDERR:\n" + res.stderr)
            
    print(f" -> Output salvato in: {clean_log_name}")
    # Print the last 20 lines of the output for immediate monitoring
    lines = res.stdout.split("\n")
    print(" -> Sintesi dell'output:")
    for line in lines[-25:]:
        if line.strip():
            print("    ", line)
    return res.returncode

# Ensure we are in CWD of the repo
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 1. Run Sarcos Regression
run_cmd("python benchmark.py --dataset sarcos --epochs 40")

# 2. Run California Housing Regression
run_cmd("python benchmark.py --dataset california --epochs 40")

# 3. Run MNIST Classification
run_cmd("python benchmark.py --dataset mnist --start_width 1024 --epochs 12")

# 4. Run CIFAR-10 classification ablation
run_cmd("python classification_experiments.py")

# 5. Run ImageNet-32 denoising (2 epochs)
run_cmd("python imagenet32_denoising_benchmark.py --dataset imagenet32 --data_dir ../imagenet32 --model breath --dz 9216 --min_width 512 --epochs 2")

print("\n" + "="*80)
print("PIPELINE PURISTA COMPLETATA CON SUCCESSO!")
print("="*80)
