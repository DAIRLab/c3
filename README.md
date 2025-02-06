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
