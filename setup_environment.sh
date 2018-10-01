#!/bin/bash
source ~/.bashrc
# Verify Conda installation: (https://conda.io/docs/user-guide/install/index.html)
# return 1 if global command line program installed, else 0
# example
# echo "node: $(program_is_installed node)"
if [[ "$OSTYPE" =~ ^darwin ]]; then
    echo "Automated installation on MacOS not supported."
    echo "Please read setup_environment.sh and install manually."
    exit 1
fi

function program_is_installed {
  # set to 1 initially
  local return_=1
  # set to 0 if not found
  type $1 >/dev/null 2>&1 || { local return_=0; }
  # return value
  echo "$return_"
}


if [ $(program_is_installed conda) == 1 ]; then
  echo "conda installed"
  echo 'source ~/.bash_functions' >> ~/.bashrc
else
  echo "installing conda"
  wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
  chmod +x Miniconda3-latest-Linux-x86_64.sh
  ./Miniconda3-latest-Linux-x86_64.sh
  tail -n 1 ~/.bashrc >> ~/.bash_functions
  head -n -2 ~/.bashrc
  echo 'source ~/.bash_functions' >> ~/.bashrc
  source ~/.bash_functions

fi

# Create an environment
conda env create -f environment.yml

# Set up PYTHONPATH
touch ~/.bash_functions
echo 'export PYTHONPATH=$PWD:$PYTHONPATH' >> ~/.bash_functions

# Congratulate user on success
echo "You're the best! Everything worked!"
