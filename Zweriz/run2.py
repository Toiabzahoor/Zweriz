import modal
import subprocess

# Stripped out Rust and build-essential installations because the binary is already compiled.
# This will make your container spin up significantly faster.
image = (
    modal.Image.from_registry("nvidia/cuda:12.2.2-devel-ubuntu22.04", add_python="3.11")
    # Pushing the local directory. Ensure "bin" is NOT in the ignore list so the executable goes up.
    .add_local_dir(".", remote_path="/root/app", ignore=["target", ".git"]) 
)

app = modal.App("gpu-lang-engine")

@app.function(image=image, gpu="T4") 
def run_binary():
    print("Executing compiled Zweriz CUDA binary on NVIDIA node...")
    
    subprocess.run(
        [
            "/root/app/bin/zweriz_cuda", # Execute the compiled binary directly
            "ui.zw"                      # Pass the script you want to run
        ], 
        cwd="/root/app/chess",           # Execute from inside the chess folder
        check=True
    )

@app.local_entrypoint()
def main():
    run_binary.remote()