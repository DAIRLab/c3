# c3
Consensus Complementarity Control

This is a standalone repository for the C3 algorithm. For more in-depth examples, see dairlib.

Currently, it is only confirmed that this process works on Ubuntu 20.04 with Python 3.8! 

## Installation
1. Install Drake via apt, following instructions https://drake.mit.edu/apt.html#stable-releases
2. Download and extract Gurobi 9.5.0: https://packages.gurobi.com/9.5/gurobi9.5.0_linux64.tar.gz
3. Install pyc3
```
pip install https://www.seas.upenn.edu/~posa/files/pyc3-0.0.1-py3-none-linux_x86_64.whl
```
4. Set LD_LIBRARY_PATH
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/drake/lib/:<GUROBI_LOCATION>/linux64/lib
```

## Example
See `src/test/c3_py_ti_test.py`