import sys
import subprocess
import importlib.metadata

def check_and_install_requirements(req_file="requirements.txt"):
    print(f"Checking dependencies from {req_file}...")
    try:
        with open(req_file, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: {req_file} not found.")
        return

    packages_to_check = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        
        # Remove inline comments and get the package string
        pkg_str = line.split("#")[0].strip()
        packages_to_check.append(pkg_str)

    missing_packages = []
    for pkg_str in packages_to_check:
        # Extract the base package name (e.g., langchain-core from langchain-core==1.4.0)
        pkg_name = pkg_str.split("==")[0].split(">=")[0].split("<=")[0].split("~=")[0].strip()
        try:
            # Check if package is installed
            importlib.metadata.version(pkg_name)
        except importlib.metadata.PackageNotFoundError:
            missing_packages.append(pkg_str)

    if not missing_packages:
        print("All required packages are already installed. System is ready!")
    else:
        print(f"Missing packages found: {len(missing_packages)}")
        for pkg in missing_packages:
            print(f" - {pkg}")
            
        print("\nInstalling missing packages...")
        try:
            # Using pip install -r requirements.txt ensures all dependencies and versions are properly resolved
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])
            print("\nSuccessfully installed missing packages!")
        except subprocess.CalledProcessError as e:
            print(f"\nFailed to install packages. Error: {e}")
            sys.exit(1)
if __name__ == "__main__":
    check_and_install_requirements()
