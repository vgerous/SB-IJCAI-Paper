# Might need to configure Annaconda for Non-DS OS image
# https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh

# In conda environment
conda create --name SpotBlock python=3.8
conda activate SpotBlock
conda install -c conda-forge cvxpy fbprophet pmdarima

# For GPU machine
conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch
# For CPU machine
conda install pytorch torchvision torchaudio cpuonly -c pytorch

conda install -y -c conda-forge scikit-learn=0.22.2

pip install numpy pandas qpth statsmodels sklearn openpyxl xlwt

# Maybe need some other alternative if we still have license issue
pip install -i https://pypi.gurobi.com gurobipy

# Download and unpack Gurobi
mkdir ~/pkgs
cd ~/pkgs
wget https://packages.gurobi.com/9.1/gurobi9.1.1_linux64.tar.gz
tar xzvf gurobi9.1.1_linux64.tar.gz
# Apply a key in Account setting
~/pkgs/gurobi911/linux64/bin/grbgetkey 00000000-0000-0000-0000-000000000000

# Get source
# Need to generate/upload SSH keys
mkdir ~/source
cd ~/source
git clone git@ssh.dev.azure.com:v3/chual/SpotBlockResearch/Competitors
