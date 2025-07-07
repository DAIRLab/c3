# C3
Consensus Complementarity Control

This is a standalone repository for the C3 algorithm. For more in-depth examples, see dairlib. 
Currently we only officially support Ubuntu 22.04. 

## Build from source
1. Clone C3 (Don't change to the c3 directory yet)
```shell
git clone --filter=blob:none git@github.com:DAIRLab/c3.git 
```

2. Install Drake's dependencies by running the commands below. This will download the specific Drake release used by C3, and install the corresponding dependencies. 
```shell
git clone --depth 1 --branch v1.35.0 https://github.com/RobotLocomotion/drake.git
sudo drake/setup/ubuntu/install_prereqs.sh
```
3. Follow the instructions at https://drake.mit.edu/bazel.html to install Gurobi 10.0
4. Change to the C3 Directory, and build the repo:
```shell
cd c3
bazel build ...
```
5. Run an example program
```shell
bazel-bin/bindings/test/c3_py_test
```
6. You may delete the copy of Drake we cloned in step 3


# Analyze the improved C3
analyze.py plots the delta value before and after the projection step. There are several parameters:
1. projection: 0 for z output before projection; 1 for delta output after projection (default 0)
2. plot_horizon: True if you wish to plot the variable W.R.T to the horizon; False if you wish t plot the variable W.R.T the ADMM iteration (default False)
3. plot_dual: True if you wish to plot the gamma V.S. lambda plots; False for otherwise (default False)
4. horizon: an integer that chooses which horizon step you are interested in
5. timestep: an integer that chooses which timestep you are interested in
6. choose_admm: an integer that chooses which ADMM step you are interested in (only needed when plot_dual is True)
7. original: True if you want to investigate original ADMM or planned horizon iterations
8. admm_num: the number of total ADMM iterations (default 10)