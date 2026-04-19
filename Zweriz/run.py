import modal
import subprocess
import os

image = (
    modal.Image.from_registry("nvidia/cuda:12.2.2-devel-ubuntu22.04", add_python="3.11")
    .apt_install("curl", "build-essential")
    .run_commands("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y")
    .add_local_dir(".", remote_path="/root/app", ignore=["target", ".git"]) 
)

app = modal.App("gpu-lang-engine")

@app.function(image=image, gpu="T4") 
def run_compiler():
    print("Executing on NVIDIA CUDA node...")
    
    subprocess.run(
        [
            "/root/.cargo/bin/cargo", 
            "run", 
            "--manifest-path", "/root/app/Cargo.toml", # Explicitly point Cargo to the root manifest
            "--release", 
            "--features", "cuda",
            "--", 
            "ui.zw" # Run the script by its local name
        ], 
        cwd="/root/app/chess", # <--- FIX: Execute from inside the chess folder
        check=True
    )

@app.local_entrypoint()
def main():
    run_compiler.remote()