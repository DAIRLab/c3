# c3
Consensus Complementarity Control

This is a standalone repository for the C3 algorithm. For more in-depth examples, see dairlib.

## Installation
1. Install Drake via apt: https://drake.mit.edu/apt.html#stable-releases
2. Download and extract Gurobi 9.5.0: https://packages.gurobi.com/9.5/gurobi9.5.0_linux64.tar.gz
3. Install pyc3
```
pip install pyc3
```
4. Set LD_LIBRARY_PATH
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/drake/bin/:<GUROBI_LOCATION>/linux64/lib
```