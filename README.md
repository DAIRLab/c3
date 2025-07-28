# C3: Consensus Complementarity Control

[**Main Paper: Consensus Complementarity Control**](https://arxiv.org/abs/2304.11259)

<p align="center">
    <a href="https://cirrus-ci.com/github/DAIRLab/c3">
        <strong style="vertical-align: middle;">Build (Noble)</strong>
        <img src="https://api.cirrus-ci.com/github/DAIRLab/c3.svg?task=noble&script=test&branch=main" alt="Noble Test Report" style="vertical-align: middle;"/>
    </a>
    &nbsp;&nbsp;
    <a href="https://cirrus-ci.com/github/DAIRLab/c3">
        <strong style="vertical-align: middle;">Build (Jammy)</strong>
        <img src="https://api.cirrus-ci.com/github/DAIRLab/c3.svg?task=jammy&script=test&branch=main" alt="Jammy Test" style="vertical-align: middle;"/>
    </a>
    &nbsp;&nbsp;
    <a href="https://github.com/DAIRLab/c3/actions/workflows/coverage.yml"><strong style="vertical-align: middle;">Coverage</strong>
        <img src="https://github.com/DAIRLab/c3/actions/workflows/coverage.yml/badge.svg?branch=main&event=push" alt="C3 Coverage" style="vertical-align: middle;"/>
    </a>
</p>

This repository contains the reference implementation of the [Consensus Complementarity Control (C3)](https://arxiv.org/abs/2304.11259) algorithm. For more in-depth examples, see [dairlib](https://github.com/DAIRLab/dairlib).  
**Officially supported OS:** Ubuntu 22.04.

---

## Table of Contents

- [Setup](#setup)
- [Build Instructions](#build-instructions)
- [Testing & Coverage](#testing-and-coverage)
- [Running Examples](#running-examples)
- [Directory Structure](#directory-structure)
- [Reference](#reference)

---

## Setup

1. **Install Bazel or Bazelisk:**  
    You can install Bazelisk (a user-friendly launcher for Bazel) or Bazel directly. Bazelisk is recommended as it automatically manages Bazel versions.

    **To install Bazelisk:**
    ```sh
    sudo apt-get update
    sudo apt-get install -y curl
    sudo curl -L https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64 -o /usr/local/bin/bazel
    sudo chmod +x /usr/local/bin/bazel
    ```
    For more details and to find a specific version, visit the [Bazelisk releases page](https://github.com/bazelbuild/bazelisk/releases). Choose a version compatible with your system and project requirements.

    **Or, to install Bazel directly:**  
    Follow the instructions at [Bazel's official installation guide](https://bazel.build/install/ubuntu).
2. **Clone C3 (do not `cd` into the directory yet):**
    ```sh
    git clone --filter=blob:none git@github.com:DAIRLab/c3.git
    ```

2. **Install Drake and its dependencies:**
    ```sh
    git clone --depth 1 --branch v1.35.0 https://github.com/RobotLocomotion/drake.git
    sudo drake/setup/ubuntu/install_prereqs.sh
    ```

3. **Install Gurobi 10.0:**  
   Follow the instructions at [Drake's Gurobi setup page](https://drake.mit.edu/bazel.html) to install Gurobi 10.0.

4. **(Optional) Remove Drake clone:**  
   You may delete the Drake directory after installing dependencies.

---

## Build Instructions

1. **Change to the C3 directory:**
    ```sh
    cd c3
    ```

2. **Build the repository using Bazel:**
    ```sh
    bazel build ...
    ```

---

## Testing and Coverage

- **Run all unit tests:**
    ```sh
    bazel test ... --test_output=all
    ```

- **Run a specific test:**
    ```sh
    bazel test //systems:systems_test
    ```

- **Run coverage:**
    ```sh
    bazel coverage --combined_report=lcov ...
    genhtml --branch-coverage --output genhtml "$(bazel info output_path)/_coverage/_coverage_report.dat" # Generates the HTML report to be viewed inside the genhtml folder
    ```

## Running Examples

This repository provides several C++ and Python examples demonstrating how to use the C3 library for different systems and workflows.

---

### Quick Start

- **C++ Examples:**  
  See the [C3 Standalone Example](./examples/README.md#c3-standalone-example), [C3 Controller Example](./examples/README.md#c3-controller-example), and [LCS Factory System Example](./examples/README.md#lcs-factory-system-example) sections in the examples README for build and run instructions.

- **Python Examples:**  
  See the [Python Examples](./examples/README.md#python-examples) section in the examples README for how to build and run the Python scripts using Bazel.

---

For more information, including how to select different problems, visualize results, and understand the system architectures, refer to the [`examples/README.md`](./examples/README.md) file.

---

## Directory Structure

```plaintext
c3/
├── bindings/          # Python bindings and tests
├── core/              # Core algorithm implementation
├── examples/          # Example applications and simulations
├── systems/           # Drake systems and tests
├── multibody/         # algorithms for computation in multibody environments
├── third_party/       # External dependencies
└── MODULE.bazel       # Bazel module file
```
---

## Reference

For a detailed explanation of the C3 algorithm, please refer to the [main paper](https://arxiv.org/abs/2304.11259). Additional resources and in-depth examples can be found in the [dairlib repository](https://github.com/DAIRLab/dairlib).

## Citation
If you use C3 in your research, please cite:

```bibtex
@misc{aydinoglu2024consensuscomplementaritycontrolmulticontact,
      title={Consensus Complementarity Control for Multi-Contact MPC}, 
      author={Alp Aydinoglu and Adam Wei and Wei-Cheng Huang and Michael Posa},
      year={2024},
      eprint={2304.11259},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2304.11259}, 
}
```
