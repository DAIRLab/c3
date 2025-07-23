# pyc3 Python Bindings

This directory contains the Bazel build configuration and source files for the `pyc3` Python bindings for C3 (Consensus Complementarity Control).

The `pyc3` package provides Python bindings for the C3 algorithm, allowing you to work with Linear Complementarity Systems (LCS), solve Mixed-Integer Quadratic Programs (MIQP), and control systems using the C3 framework.

---

## Prerequisites

Before building and installing pyc3, ensure you have:

1. **Drake**: C3 depends on Drake for multibody dynamics and optimization
2. **Gurobi**: Required for the MIQP solver (version 10.0 recommended)
3. **Bazel**: For building the project
4. **Python 3.8+**: With development headers

For detailed setup instructions, see the main [C3 README](../../README.md).

---

## Building and Installing pyc3 in a Virtual Environment

Follow these steps to build the `pyc3` Python wheel and install it into a virtual environment.

### 1. Create and Activate a Python Virtual Environment

```sh
python3 -m venv venv
source venv/bin/activate
```

### 2. Build the pyc3 Python Wheel

From the root of the C3 repository, build the Python wheel using Bazel:

```sh
bazel build //bindings/pyc3:pyc3_wheel
```

This will create a wheel file in the Bazel output directory.

### 3. Install the pyc3 Wheel

Install the built wheel into your virtual environment:

```sh
pip install bazel-bin/bindings/pyc3/pyc3_wheel.whl
```

### 4. Verify Installation

Test that pyc3 was installed correctly:

```python
python -c 'import pyc3; print("pyc3 installed successfully!")'
```

---

## Examples

Complete examples are available in the `examples/python/` directory:

- **`c3_example.py`**: Basic C3 optimization example
- **`c3_controller_example.py`**: Real-time controller with Drake integration
- **`lcs_factory_example.py`**: LCS creation from multibody systems

To run an example:

```sh
cd examples/python
python c3_example.py
```

---

## Troubleshooting

### Common Issues

1. **Import Error**: If you get import errors, ensure Drake is properly installed and your virtual environment is activated.

2. **Gurobi License**: Make sure your Gurobi license is properly configured:
   ```sh
   export GUROBI_HOME=/opt/gurobi1000/linux64
   export GRB_LICENSE_FILE=/path/to/your/gurobi.lic
   ```

3. **Build Errors**: Ensure all prerequisites are installed and Bazel can find Drake:
   ```sh
   bazel clean
   bazel build //bindings/pyc3:pyc3_wheel
   ```

### Getting Help

- Check the main [C3 repository issues](https://github.com/DAIRLab/c3/issues)
- Review the [C3 paper](https://arxiv.org/abs/2304.11259) for algorithmic details
- See [dairlib](https://github.com/DAIRLab/dairlib) for more comprehensive examples

