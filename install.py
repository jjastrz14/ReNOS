#!/usr/bin/env python
import os
import sys
import subprocess
import argparse
from pathlib import Path
import platform

# Define package requirements (excluding TensorFlow which will be handled separately)
REQUIREMENTS = [
    # Core dependencies
    "keras==2.12.0",       # Matching TensorFlow version
    "networkx>=2.8.0",     # For graph operations
    "numpy>=1.22.0",       # Numerical computing
    "matplotlib>=3.5.0",   # Plotting utilities
    "pytz",                # Timezone support
    "prettytable",         # Pretty table formatting
    "pydot",              # For parallel processing
    "pybind11"
    
    # Domain-specific packages
    "larq>=0.12.2",        # Binarized Neural Networks library
    
    # Optional dependencies for development
    "jupyter",             # For notebook support
    "pytest",              # For testing
]

# Additional packages that might require special installation
SPECIAL_PACKAGES = {
    # These might need separate installation commands or have specific versions
    "onnx": "onnx>=1.12.0",
    "onnxruntime": "onnxruntime>=1.12.0",
    "skl2onnx": "skl2onnx>=1.12.0",
    "onnxconverter-common": "onnxconverter-common>=1.12.0"
}

def run_command(cmd, verbose=True):
    """Run a shell command and optionally print output"""
    if verbose:
        print(f"Running: {' '.join(cmd)}")
    
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    stdout, stderr = process.communicate()
    
    if verbose:
        if stdout:
            print(stdout)
        if stderr:
            print(f"Error: {stderr}")
    
    return process.returncode, stdout, stderr

def check_conda():
    """Check if conda is installed and available"""
    try:
        subprocess.run(["conda", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def is_package_installed(package_name, env_name=None):
    """Check if a package is already installed"""
    package_base = package_name.split('==')[0].split('>=')[0].split('<=')[0].strip()
    
    if env_name:
        # Check in conda environment
        if platform.system() == "Windows":
            python_cmd = f"conda run -n {env_name} python"
        else:
            python_cmd = f"conda run --live-stream -n {env_name} python"
        
        cmd = python_cmd.split() + ["-c", f"import pkg_resources; print(pkg_resources.get_distribution('{package_base}').version)"]
    else:
        # Check in current environment
        cmd = [sys.executable, "-c", f"import pkg_resources; print(pkg_resources.get_distribution('{package_base}').version)"]
    
    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        stdout, _ = process.communicate()
        
        if process.returncode == 0 and stdout.strip():
            installed_version = stdout.strip()
            print(f"Found {package_base} {installed_version}")
            
            # If specific version is required, check if it matches
            if '==' in package_name:
                required_version = package_name.split('==')[1]
                if installed_version == required_version:
                    return True
                else:
                    print(f"Version mismatch: found {installed_version}, required {required_version}")
                    return False
            return True
        return False
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def create_conda_env(env_name, python_version="3.9"):
    """Create a new conda environment"""
    print(f"Creating new conda environment '{env_name}' with Python {python_version}...")
    cmd = ["conda", "create", "-n", env_name, f"python={python_version}", "-y"]
    return run_command(cmd)

def install_tensorflow_conda(env_name):
    """Install TensorFlow using conda-forge"""
    # Check if already installed
    if is_package_installed("tensorflow", env_name):
        # Check version
        if platform.system() == "Windows":
            python_cmd = f"conda run -n {env_name} python"
        else:
            python_cmd = f"conda run --live-stream -n {env_name} python"
        
        cmd = python_cmd.split() + ["-c", "import tensorflow as tf; print(tf.__version__)"]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, _ = process.communicate()
        
        if process.returncode == 0 and stdout.strip().startswith("2.12"):
            print(f"TensorFlow 2.12 is already installed in environment '{env_name}'")
            return 0, "Already installed", ""
    
    print(f"Installing TensorFlow 2.12 from conda-forge in environment '{env_name}'...")
    cmd = ["conda", "install", "-n", env_name, "-c", "conda-forge", "-y", "tensorflow=2.12"]
    return run_command(cmd)

def install_tensorflow_pip(env_name=None):
    """Install TensorFlow using pip"""
    # Check if already installed
    if is_package_installed("tensorflow==2.12.0", env_name):
        print("TensorFlow 2.12.0 is already installed")
        return 0, "Already installed", ""
    
    print("Installing TensorFlow 2.12.0 with pip...")
    if env_name:
        if platform.system() == "Windows":
            python_cmd = f"conda run -n {env_name} python"
        else:
            python_cmd = f"conda run --live-stream -n {env_name} python"
        
        cmd = python_cmd.split() + ["-m", "pip", "install", "tensorflow==2.12.0"]
    else:
        cmd = [sys.executable, "-m", "pip", "install", "tensorflow==2.12.0"]
    
    return run_command(cmd)

def install_requirements(env_name=None):
    """Install remaining requirements using pip"""
    print("Checking and installing remaining packages...")
    pip_packages = REQUIREMENTS + list(SPECIAL_PACKAGES.values())
    
    # Filter out already installed packages if not in a conda env
    if not env_name:
        packages_to_install = []
        for package in pip_packages:
            if not is_package_installed(package):
                packages_to_install.append(package)
        
        if not packages_to_install:
            print("All packages are already installed.")
            return True
        
        print(f"Installing {len(packages_to_install)} packages...")
        cmd = [sys.executable, "-m", "pip", "install"] + packages_to_install
        code, _, stderr = run_command(cmd)
        if code != 0:
            print("\nPip installation failed. Error:")
            print(stderr)
            return False
    else:
        success = pip_install_in_conda(env_name, pip_packages)
        if not success:
            return False
    
    return True

def pip_install_in_conda(env_name, packages):
    """Install packages with pip within a conda environment"""
    # Filter out already installed packages
    packages_to_install = []
    for package in packages:
        if not is_package_installed(package, env_name):
            packages_to_install.append(package)
    
    if not packages_to_install:
        print("All packages are already installed.")
        return True
    
    print(f"Installing {len(packages_to_install)} packages...")
    
    if platform.system() == "Windows":
        python_cmd = f"conda run -n {env_name} python"
    else:
        python_cmd = f"conda run --live-stream -n {env_name} python"
    
    cmd = python_cmd.split() + ["-m", "pip", "install"] + packages_to_install
    code, _, stderr = run_command(cmd)
    
    if code != 0:
        print("\nPip installation failed in conda environment. Error:")
        print(stderr)
        return False
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Install dependencies for the project")
    parser.add_argument("--create-env", action="store_true", help="Create a new conda environment")
    parser.add_argument("--env-name", default="renos", help="Name of the conda environment to create or use")
    parser.add_argument("--python-version", default="3.11.11", help="Python version to use (3.11.11 recommended for TensorFlow 2.12)")
    parser.add_argument("--tensorflow-conda", action="store_true", help="Install TensorFlow from conda-forge")
    
    args = parser.parse_args()
    
    # Check if conda is available
    conda_available = check_conda()
    
    if args.create_env:
        if not conda_available:
            print("Error: conda is not available. Cannot create a new environment.")
            sys.exit(1)
        
        # Create the conda environment
        code, _, _ = create_conda_env(args.env_name, args.python_version)
        if code != 0:
            print(f"Failed to create conda environment '{args.env_name}'")
            sys.exit(1)
    
    # Install TensorFlow
    if conda_available and args.tensorflow_conda:
        # Install TensorFlow via conda-forge
        code, _, _ = install_tensorflow_conda(args.env_name)
        if code != 0:
            print("Failed to install TensorFlow from conda-forge")
            print("Falling back to pip installation...")
            code, _, _ = install_tensorflow_pip(args.env_name if conda_available else None)
            if code != 0:
                print("Failed to install TensorFlow with pip")
                sys.exit(1)
    else:
        # Install TensorFlow via pip
        code, _, _ = install_tensorflow_pip(args.env_name if conda_available else None)
        if code != 0:
            print("Failed to install TensorFlow with pip")
            sys.exit(1)
    
    # Install remaining requirements
    success = install_requirements(args.env_name if conda_available else None)
    
    if success:
        print("\nInstallation completed successfully!")
        
        if args.create_env and conda_available:
            print(f"\nTo activate the conda environment, run:")
            print(f"    conda activate {args.env_name}")
    else:
        print("\nInstallation failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    print("=== Project Dependencies Installation ===")
    
    # Ask user for installation preference if running interactively
    if sys.stdout.isatty():
        conda_available = check_conda()
        
        if conda_available:
            create_new_env = input("Do you want to create a new conda environment? (y/n): ").lower() == 'y'
            
            if create_new_env:
                env_name = input("Enter environment name (default: renos): ") or "renos"
                print("\nNOTE: TensorFlow 2.12 works best with Python 3.11.11 or earlier")
                python_version = input("Enter Python version (default: 3.11): ") or "3.11"
                
                # Create conda environment
                code, _, _ = create_conda_env(env_name, python_version)
                if code != 0:
                    print(f"Failed to create conda environment '{env_name}'")
                    sys.exit(1)
                
                # Ask about TensorFlow installation method
                use_conda_tf = input("Install TensorFlow from conda-forge? (recommended) (y/n): ").lower() == 'y'
                
                if use_conda_tf:
                    code, _, _ = install_tensorflow_conda(env_name)
                    if code != 0:
                        print("Failed to install TensorFlow from conda-forge")
                        fallback = input("Fall back to pip installation for TensorFlow? (y/n): ").lower() == 'y'
                        if fallback:
                            code, _, _ = install_tensorflow_pip(env_name)
                            if code != 0:
                                print("Failed to install TensorFlow with pip")
                                sys.exit(1)
                        else:
                            print("Installation aborted.")
                            sys.exit(1)
                else:
                    code, _, _ = install_tensorflow_pip(env_name)
                    if code != 0:
                        print("Failed to install TensorFlow with pip")
                        sys.exit(1)
                
                # Install remaining packages
                success = install_requirements(env_name)
                if not success:
                    print("Installation of remaining packages failed.")
                    sys.exit(1)
                
            else:
                use_existing_env = input("Install in an existing conda environment? (y/n): ").lower() == 'y'
                
                if use_existing_env:
                    env_name = input("Enter environment name: ")
                    
                    # Get Python version of the environment
                    cmd = ["conda", "run", "-n", env_name, "python", "--version"]
                    _, stdout, _ = run_command(cmd, verbose=False)
                    if stdout:
                        python_version = stdout.strip().split()[1]
                        if  python_version.startswith("3.11") or python_version.startswith("3.12") or python_version.startswith("3.13"):
                            print(f"\nWARNING: Your environment uses Python {python_version}")
                            print("TensorFlow 2.12 is likely NOT compatible with this Python version.")
                            print("We recommend using Python 3.11.11 or earlier for TensorFlow 2.12")
                            proceed = input("Do you want to proceed anyway? (y/n): ").lower() == 'y'
                            if not proceed:
                                print("Installation aborted.")
                                sys.exit(0)
                    
                    # Ask about TensorFlow installation method
                    use_conda_tf = input("Install TensorFlow from conda-forge? (recommended) (y/n): ").lower() == 'y'
                    
                    if use_conda_tf:
                        code, _, _ = install_tensorflow_conda(env_name)
                        if code != 0:
                            print("Failed to install TensorFlow from conda-forge")
                            fallback = input("Fall back to pip installation for TensorFlow? (y/n): ").lower() == 'y'
                            if fallback:
                                code, _, _ = install_tensorflow_pip(env_name)
                                if code != 0:
                                    print("Failed to install TensorFlow with pip")
                                    sys.exit(1)
                            else:
                                print("Installation aborted.")
                                sys.exit(1)
                    else:
                        code, _, _ = install_tensorflow_pip(env_name)
                        if code != 0:
                            print("Failed to install TensorFlow with pip")
                            sys.exit(1)
                    
                    # Install remaining packages
                    success = install_requirements(env_name)
                    if not success:
                        print("Installation of remaining packages failed.")
                        sys.exit(1)
                    
                else:
                    print("Installing packages in the current Python environment")
                    print("WARNING: TensorFlow 2.12 may not be compatible with your Python version")
                    
                    # Ask about TensorFlow installation method
                    use_conda_tf = False  # Can't use conda without an environment
                    
                    code, _, _ = install_tensorflow_pip()
                    if code != 0:
                        print("Failed to install TensorFlow with pip")
                        sys.exit(1)
                    
                    # Install remaining packages
                    success = install_requirements()
                    if not success:
                        print("Installation of remaining packages failed.")
                        sys.exit(1)
        else:
            print("Conda is not available. Using pip for installation.")
            print("WARNING: TensorFlow 2.12 may not be compatible with your Python version")
            
            code, _, _ = install_tensorflow_pip()
            if code != 0:
                print("Failed to install TensorFlow with pip")
                sys.exit(1)
            
            success = install_requirements()
            if not success:
                print("Installation of remaining packages failed.")
                sys.exit(1)
        
        print("\nInstallation completed successfully!")
    else:
        # Non-interactive mode, use arguments
        main()