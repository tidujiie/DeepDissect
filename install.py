import os
import platform
import subprocess

def run_command(command):
    try:
        subprocess.run(command, shell=True, check=True, executable="/bin/bash")
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        exit(1)

def get_conda_activate_cmd(env_name):
    """ Returns the correct conda activation command for Linux """
    conda_sh_path = subprocess.run("conda info --base", shell=True, check=True, capture_output=True, text=True).stdout.strip()
    return f"source {conda_sh_path}/etc/profile.d/conda.sh && conda activate {env_name} && "

def main():
    print("Cuda 10.2 is required for this installation")
    env_name = input("Enter the name of the Conda environment: ").strip()
    
    print(f"Creating Conda environment: {env_name}")
    run_command(f"conda create --name {env_name} python=3.8 -y")

    activate_cmd = get_conda_activate_cmd(env_name)

    # Common packages
    run_command(activate_cmd + "pip install -U openmim")
    run_command(activate_cmd + "pip install torch==1.10.0+cu102 torchvision==0.11.0+cu102 torchaudio==0.10.0 --extra-index-url https://download.pytorch.org/whl/cu102")
    run_command(activate_cmd + "pip install -U openmim")
    run_command(activate_cmd + "mim install mmengine")
    run_command(activate_cmd + "mim install 'mmcv==2.0.1'")
    run_command(activate_cmd + "mim install mmdet")
    
    print(f"Installation complete. Environment {env_name} is ready.")
    
    # Install deep_dissect library
    print("Installing deep_dissect library...")
    run_command(activate_cmd + "pip install setuptools wheel")
    run_command(activate_cmd + "pip install .")
    
if __name__ == "__main__":
    main()
