# How to Integrate the C3 Library into an Existing Project

This guide provides step-by-step instructions for integrating the Contact-Implicit Model Predictive Control (C3) library into an existing C++ project built with Bazel. It draws from real-world integration examples and includes detailed parameter documentation.

## Table of Contents
1. [Dependency Management Setup](#dependency-management-setup)
2. [BUILD.bazel Configuration](#buildbazel-configuration)
3. [Code Integration Patterns](#code-integration-patterns)
4. [Configuration and Parameters](#configuration-and-parameters)
   - [LCSFactoryOptions](#lcsfactoryoptions)
   - [C3Options](#c3options)
   - [C3ControllerOptions](#c3controlleroptions)

---

## Dependency Management Setup

### WORKSPACE Approach (Legacy)

If your project uses **WORKSPACE**, add C3 as an external repository:

```python
# WORKSPACE
bazel_dep(name = "c3")
git_override(
    module_name = "c3",
    remote = "https://github.com/DAIRLab/c3.git",
    commit = "f0e1358bbf0413ef655ffb6bc3be556b07c6bfd5"
)

```

Or using an archive:

```python
http_archive(
    name = "c3",
    url = "https://github.com/DAIRLab/c3/archive/refs/heads/main.zip",
    strip_prefix = "c3-main",
)
```

### Verification

After setup, test that the dependency is available:
```bash
bazel query "@c3//:libc3"
```

---

## BUILD.bazel Configuration

### Basic Dependency Declaration

To use C3 in a target, add the C3 library to your `deps`:

```python
cc_library(
    name = "my_c3_controller",
    srcs = ["my_c3_controller.cc"],
    hdrs = ["my_c3_controller.h"],
    deps = [
        "@c3//:libc3",  # Main C3 library
        # Your other dependencies
    ],
)
```

### Common C3 Targets

The C3 library exposes the following main targets:

| Target | Usage |
|--------|-------|
| `@c3//:libc3` | Core C3 solver and multibody utilities |
| `@c3//lcmtypes:lcmt_c3` | LCM message types for C3 (optional, if using LCM) |

### Binary Configuration Example

Here's a complete example of a C3-enabled binary:

```python
cc_binary(
    name = "run_c3_controller",
    srcs = ["c3_controller_main.cc"],
    data = [
        # Config files needed at runtime
        "//config:c3_config_files",
        "@drake_models//:franka_description",  # If using a robot
    ],
    deps = [
        ":c3_controller_lib",
        ":parameters",  # See Configuration section
        "//systems:robot_lcm_systems",
        "//systems/controllers",
        "@c3//:libc3",
        "@c3//lcmtypes:lcmt_c3",  # For LCM integration
        "@drake//:drake_shared_library",
        "@gflags",  # For command-line configuration
    ],
)
```

---

## Code Integration Patterns

### 1. Basic Includes

Include C3 headers in your source files:

```cpp
#include <c3/core/c3.h>
#include <c3/core/c3_options.h>
#include <c3/core/lcs.h>
#include <c3/core/solver_options_io.h>
#include <c3/multibody/lcs_factory.h>
#include <c3/multibody/lcs_factory_options.h>
#include <c3/systems/c3_controller.h>
#include <c3/systems/lcs_factory_system.h>
```

**Note:** The exact headers depend on which C3 features you use. Additional variant-specific headers are available:
- `c3/core/c3_qp.h` - QP formulation
- `c3/core/c3_miqp.h` - Mixed-integer QP formulation  
- `c3/core/c3_plus.h` - C3+ projection variant

### 2. Initialization Pattern

Most C3 integrations follow this initialization sequence:

```cpp
#include <drake/common/yaml/yaml_io.h>

// 1. Load LCS factory options from YAML
auto lcs_options = drake::yaml::LoadYamlFile<
    c3::multibody::LCSFactoryOptions>(
    drake::FindResourceOrThrow("file:///path/to/lcs_factory_options.yaml"));

// 2. Create LCS factory instance
c3::multibody::LCSFactory lcs_factory(plant, context, plant_ad, context_ad, lcs_options);

// 3. Update state and input
lcs_factory.UpdateStateAndInput(state, input);

// 4. Generate LCS
auto lcs = lcs_factory.GenerateLCS();

// 5. Load C3 solver options from YAML
auto c3_options = drake::yaml::LoadYamlFile<
    c3::C3Options>(
    drake::FindResourceOrThrow("file:///path/to/c3_options.yaml"));

// 6. Create C3 solver instance
c3::C3 c3_solver(lcs, c3::C3::CreateCostMatricesFromC3Options(c3_options, lcs_options.N),
                 x_desired, c3_options);
```

### 3. Using C3 in a Drake System

Integrate C3 into your Drake system diagram:

```cpp
#include <drake/common/yaml/yaml_io.h>
#include "systems/c3_controller.h"
#include "systems/c3_controller_options.h"
#include "systems/lcs_factory_system.h"
#include "multibody/lcs_factory.h"

drake::systems::DiagramBuilder<double> builder;

// 1. Load Options
auto controller_options = drake::yaml::LoadYamlFile<c3::systems::C3ControllerOptions>(
    "path/to/c3_controller_options.yaml");

// Create LCSFactory instance
c3::multibody::LCSFactory lcs_factory(plant, context, plant_ad, context_ad,
                                      controller_options.lcs_factory_options);

// Update state and input
lcs_factory.UpdateStateAndInput(state, input);

// Generate LCS
auto lcs = lcs_factory.GenerateLCS();

// Create C3 solver
c3::C3 c3_solver(lcs, c3::C3::CreateCostMatricesFromC3Options(
                          controller_options.c3_options, controller_options.lcs_factory_options.N),
                 x_desired, controller_options.c3_options);

// Add systems to the diagram
auto lcs_system = builder.AddSystem<c3::systems::LCSFactorySystem>(&plant, controller_options.lcs_factory_options);
auto c3_controller = builder.AddSystem<c3::systems::C3Controller>(plant, controller_options);

// Connect diagram components
builder.Connect(plant.get_output_port_state(), lcs_system->get_input_port_state());
builder.Connect(plant.get_output_port_state(), c3_controller->get_input_port_state());
builder.Connect(lcs_system->get_output_port_lcs(), c3_controller->get_input_port_lcs());
builder.Connect(c3_controller->get_output_port_control(), plant.get_input_port_actuation());
```

### 4. Namespace Usage

Use C3 namespaces to simplify code:

```cpp
using c3::C3;
using c3::C3Options;
using c3::ConstraintVariable;
using c3::LCS;
using c3::multibody::LCSFactory;
using c3::multibody::LCSFactoryOptions;
using c3::systems::C3Controller;
using c3::systems::LCSFactorySystem;
using c3::systems::lcmt_generators::C3OutputGenerator;
using c3::systems::lcmt_generators::C3TrajectoryGenerator;
using c3::systems::lcmt_generators::ContactForceGenerator;
```

---

## 5. Critical Parameter Updates

If you are migrating from an older version of C3, you must update the following fields in your `.yaml` configuration or options structs. Some parameters have been removed or replaced for better clarity and functionality.

### Removed Parameters
- **`penalize_changes_in_u_across_solves`**: Replaced by `penalize_input_change` in `C3Options`.
- **`solve_dt`**: No longer used; time step size is now defined by `dt` in `LCSFactoryOptions`.
- **`lcs_dt_resolution`**: Removed; the resolution is now implicitly handled by `dt`.
- **`dt_cost`**: Removed.
- **`use_predicted_x0`**: Replaced by `state_prediction_joints` in `C3ControllerOptions`.
- **`workspace_margins`**: Removed; workspace constraints should be explicitly defined when defining the controller.

### Updated Parameters
| Parameter                  | Type    | Description                                                                 |
|----------------------------|---------|-----------------------------------------------------------------------------|
| **`scale_lcs`**            | `bool`  | **Required.** Must be set to `true` for numerical stability in the ADMM solver. |
| **`penalize_input_change`**| `bool`  | Replaces `penalize_changes_in_u_across_solves`. Smooths control by penalizing $\Delta u$. |
| **`planar_normal_direction`** | `vector` | Required for planar models (e.g., `[0, 0, 1]`).                            |

### New Parameters in `C3ControllerOptions`
| Parameter                                | Type                | Description                                                                 |
|------------------------------------------|---------------------|-----------------------------------------------------------------------------|
| **`state_prediction_joints`**            | `vector`            | Replaces `use_predicted_x0`. Specifies joints with maximum acceleration limits for state prediction. |
| **`solve_time_filter_alpha`**            | `double`            | Low-pass filter alpha for solve time monitoring.                            |
| **`publish_frequency`**                  | `double`            | Frequency (in Hz) for publishing outputs.                                   |
| **`quaternion_weight`**                  | `double` (optional) | Scaling term for quaternion cost.                                           |
| **`quaternion_regularizer_fraction`**    | `double` (optional) | Regularization scaling term for quaternion outer product.                   |

### New Parameters in `LCSFactoryOptions`
| Parameter                                | Type                | Description                                                                 |
|------------------------------------------|---------------------|-----------------------------------------------------------------------------|
| **`contact_pair_configs`**               | `vector` (optional) | Detailed configuration for each contact pair, including friction and geometry indices. |
| **`planar_normal_direction`**            | `array<double, 3>` (optional) | Normal vector for planar contacts.                                          |
| **`planar_normal_direction_per_contact`**| `vector<array<double, 3>>` (optional) | Per-contact normal vectors for planar contacts.                             |

### Example YAML Update

Here is an example of how to update your `.yaml` configuration file:

#### Old Configuration
```yaml
c3_options:
  penalize_changes_in_u_across_solves: true
  solve_dt: 0.01
  lcs_dt_resolution: 10
  dt_cost: 0.1
  use_predicted_x0: false
  workspace_margins: 0.05
  gamma: 1.0
  rho_scale: 10
  w_Q: 50
  w_R: 25
  w_G: 0.1
  w_U: 0.5
  q_vector: [175, 175, 175, 1, 1, 1]
  r_vector: [0.15, 0.15, 0.1]
```

#### Updated Configuration
```yaml
c3_controller_options:
  projection_type: "C3+"
  solve_time_filter_alpha: 0.95
  publish_frequency: 100.0
  state_prediction_joints:
    - name: "joint_1"
      max_acceleration: 10.0
    - name: "joint_2"
      max_acceleration: 5.0
  quaternion_weight: 0.1
  quaternion_regularizer_fraction: 0.05

  c3_options:
    penalize_input_change: true
    scale_lcs: true
    gamma: 1.0
    rho_scale: 10
    w_Q: 50
    w_R: 25
    w_G: 0.1
    w_U: 0.5
    q_vector: [175, 175, 175, 1, 1, 1]
    r_vector: [0.15, 0.15, 0.1]

  lcs_factory_options:
    contact_model: "stewart_and_trinkle"
    num_contacts: 3
    dt: 0.05
    planar_normal_direction: [0, 0, 1]
    contact_pair_configs:
      - body_A: "robot_hand"
        body_B: "object"
        body_A_collision_geom_indices: [0, 1]
        body_B_collision_geom_indices: [0]
        num_friction_directions: 2
        mu: 0.5
      - body_A: "robot_hand"
        body_B: "table"
        body_A_collision_geom_indices: [1, 2]
        body_B_collision_geom_indices: [0]
        num_friction_directions: 4
        mu: 0.8
```

Ensure that all removed parameters are deleted from your configuration files, and the new parameters are added as needed.

### Handling moved contact & horizon options (migration note)

Recent C3 versions moved contact configuration and horizon/timestep fields out of `C3Options` and into the LCS factory options struct. Concretely, fields such as `N`, `dt`, `num_contacts`, `mu`, and friction-direction settings now live in `c3::multibody::LCSFactoryOptions` (or `c3::LCSFactoryOptions` depending on your include path).

What this means for you:
- Don't access contact/horizon fields from `c3::C3Options` anymore (e.g. `c3_options.N`, `c3_options.dt`, `c3_options.num_contacts`, `c3_options.num_friction_directions`, `c3_options.mu`). Those members were removed.
- Load and pass a `LCSFactoryOptions` instance to any code that needs contact/horizon configuration (the LCS factory, LCS factory system, or any helper that builds dummy LCS objects).

Minimal code patterns to follow

- Load `LCSFactoryOptions` from YAML (either the same file as your C3 options or a separate file):

```cpp
auto lcs_options = drake::yaml::LoadYamlFile<
    c3::multibody::LCSFactoryOptions>(
    drake::FindResourceOrThrow("file:///path/to/lcs_factory_options.yaml"));
```

- Construct or update components that need contact/horizon settings to accept an `LCSFactoryOptions` and use it as the canonical source:

```cpp
// Header change
explicit C3GoalGenerator(..., c3::C3Options c3_options,
                          c3::LCSFactoryOptions lcs_options, ...);

// Call site change (pass the loaded struct)
auto lcs_options = drake::yaml::LoadYamlFile<c3::LCSFactoryOptions>(...);
builder.AddSystem<C3GoalGenerator>(..., c3_options, lcs_options, ...);
```

- Use `LCSFactory` / `LCSFactoryOptions` utilities instead of manual arithmetic for contact variables/horizon sizes:

```cpp
contact_model_ = c3::multibody::GetContactModelMap().at(lcs_options.contact_model);
n_lambda_ = c3::multibody::LCSFactory::GetNumContactVariables(lcs_options);

// Use lcs_options.N and lcs_options.dt when creating dummy or time-varying
// LCS objects for port declarations or sizing.
```

Quick migration checklist
- Update YAML: add or move contact/horizon fields into the `lcs_factory_options` block (or a separate file) as shown in the examples above.
- Update includes: add `#include "c3/multibody/lcs_factory_options.h"` where needed.
- Update constructors and call sites that previously relied on `C3Options` to accept a `LCSFactoryOptions` (or otherwise obtain it) and use it for `N`, `dt`, contact configuration, and friction settings.
- Replace manual `n_lambda` and contact-model logic with `LCSFactory::GetNumContactVariables(...)` and `GetContactModelMap()`.
- Run a targeted build to catch remaining call sites or YAML fields still referencing removed members.

If you want, we can add a small automated grep to find remaining uses of the removed field names (for example: `num_contacts`, `num_friction_directions`, `N`, `dt`, `mu`) and propose fixes; this often finds places that need to be updated to use the LCS options.

---

## 6. API Signature Changes
The library has moved to a more refined namespace and class structure.

### Solver Pointer Type
The base solver class has changed. If you store a pointer to the solver in a custom class, update the type:
```cpp
// OLD
mutable std::shared_ptr<solvers::C3Base> c3_plan_;

// NEW
mutable std::shared_ptr<c3::C3> c3_plan_;
```

### LCS Initialization

Instead of manually passing matrices ($A, B, D, d$) to the solver, use the `LCSFactory` class to derive them from your `MultibodyPlant`. Note that `GenerateLCS` and `ComputeContactJacobian` are no longer static functions. You must first initialize an `LCSFactory` instance with the required options and context.

#### Example: Generating an LCS

```cpp
#include <c3/multibody/lcs_factory.h>

// Initialize LCSFactory with the plant, context, and options
c3::multibody::LCSFactory lcs_factory(plant, context, plant_ad, context_ad, options);

// Update the state and input before generating the LCS
lcs_factory.UpdateStateAndInput(state, input);

// Generate the LCS
auto lcs = lcs_factory.GenerateLCS();
```

#### Backward Compatibility

The static method `LinearizePlantToLCS` is still available for backward compatibility. However, it is recommended to use the new `LCSFactory` class for better modularity and maintainability.

#### Example: Computing Contact Jacobians

```cpp
#include <c3/multibody/lcs_factory.h>

// Initialize LCSFactory with the plant, context, and options
c3::multibody::LCSFactory lcs_factory(plant, context, plant_ad, context_ad, options);

// Compute the contact Jacobian and witness points
auto [contact_jacobian, contact_points] = lcs_factory.GetContactJacobianAndPoints();
```

This change ensures that the `LCSFactory` maintains its internal state and options, simplifying the interface for generating LCS models and computing contact-related data.

---

**Loading Configuration in C++:**

```cpp
#include <drake/common/yaml/yaml_io.h>

// Load C3 controller options (includes LCS and C3 options)
auto controller_options = drake::yaml::LoadYamlFile<
    c3::systems::C3ControllerOptions>(
    drake::FindResourceOrThrow(
        "file:///path/to/c3_controller_options.yaml"));

// Or load separately if desired
auto lcs_options = drake::yaml::LoadYamlFile<
    c3::multibody::LCSFactoryOptions>(
    drake::FindResourceOrThrow(
        "file:///path/to/lcs_factory_options.yaml"));

auto c3_opts = drake::yaml::LoadYamlFile<
    c3::C3Options>(
    drake::FindResourceOrThrow(
        "file:///path/to/c3_options.yaml"));
```

---

## Common Integration Scenarios

### Scenario 1: Plate Balancing / Manipulation Task

This is the most common scenario. Integrate C3 into a task-space controller:

**File structure:**
```
examples/plate-balancing/
├── BUILD.bazel
├── plate_balancing_c3_controller.cc
├── plate_balancing_simulation.cc
├── parameters/
│   ├── c3_scene_config.h
│   ├── plate_balancing_config.h
│   └── plate_balancing_c3_controller_options.h
├── systems/
│   └── franka_systems.h
└── config/
    └── C3/
        ├── controller-options/
        ├── scene-config/
        └── trajectory-generator-config/
```

**Key integration points:**

```cpp
// Load multibody plant with contacts
auto plant = std::make_unique<MultibodyPlant<double>>(dt);
parser.AddModelFromFile(robot_urdf, plant.get());
plant.Finalize();

// Create LCS system for contact modeling
auto lcs_system = std::make_unique<
    c3::systems::LCSFactorySystem>(plant, lcs_options);

// Create C3 controller
auto c3_controller = std::make_unique<
    c3::systems::C3Controller>(c3_solver.get());

// Connect in diagram
diagram_builder.AddSystem(std::move(lcs_system));
diagram_builder.AddSystem(std::move(c3_controller));
// ... connect ports ...
```

**BUILD.bazel for this scenario:**

```python
cc_binary(
    name = "run_c3_controller",
    srcs = ["plate_balancing_c3_controller.cc"],
    data = [":config_files", "@drake_models//:franka_description"],
    deps = [
        ":parameters",
        "//systems:robot_lcm_systems",
        "@c3//:libc3",
        "@drake//:drake_shared_library",
        "@gflags",
    ],
)
```

### Scenario 2: Sampling-Based C3 (from PR #10)

For sampling-based approaches that explore multiple futures:

**Header changes from legacy code:**

```cpp
// OLD (legacy)
#include "solvers/base_c3.h"
#include "solvers/c3_options.h"
#include "solvers/lcs.h"
#include "solvers/lcs_factory.h"
#include "solvers/solver_options_io.h"

// NEW (using C3 library)
#include "c3/core/c3.h"
#include "c3/core/c3_options.h"
#include "c3/core/lcs.h"
#include "c3/multibody/lcs_factory.h"
#include "c3/core/solver_options_io.h"
```

**BUILD.bazel migration:**

```python
# OLD (local solver)
cc_library(
    name = "sampling_c3_options",
    deps = ["//solvers:c3"]
)

# NEW (external C3 library)
cc_library(
    name = "sampling_c3_options",
    deps = ["@c3//:libc3"]
)
```

**Solver variant selection in code:**

```cpp
#include "c3/core/c3_qp.h"
#include "c3/core/c3_miqp.h"
#include "c3/core/c3_plus.h"

// Create appropriate solver variant based on configuration
std::unique_ptr<c3::C3> solver;
if (options.use_integer_formulation) {
    solver = std::make_unique<c3::C3MIQP>(lcs_options);
} else if (options.use_c3_plus) {
    solver = std::make_unique<c3::C3Plus>(lcs_options);
} else {
    solver = std::make_unique<c3::C3QP>(lcs_options);
}
```

---

## Advanced Features

### LCM Output Integration

If your project uses LCM for inter-process communication:

```cpp
#include <c3/systems/lcmt_generators/c3_output_generator.h>
#include <c3/systems/lcmt_generators/c3_trajectory_generator.h>
#include <drake/systems/lcm/lcm_publisher_system.h>

// Create C3 output generator
auto c3_output_gen = std::make_unique<
    c3::systems::lcmt_generators::C3OutputGenerator>();

// Create trajectory generator
auto traj_gen = std::make_unique<
    c3::systems::lcmt_generators::C3TrajectoryGenerator>();

// Publish via LCM
auto lcm_pub = std::make_unique<
    drake::systems::lcm::LcmPublisherSystem>(
        "C3_OUTPUT_CHANNEL");

// Connect in diagram
diagram_builder.AddSystem(std::move(c3_output_gen));
diagram_builder.AddSystem(std::move(lcm_pub));
```

**Note:** The `lcmt_generators` are only available after [PR #10](https://github.com/DAIRLab/c3/pull/10) is merged.  
![PR Status](https://img.shields.io/github/pulls/detail/state/DAIRLab/c3/10)

---

## 7. Simplified Contact Variable Calculation

In previous versions, the number of contact variables (`n_lambda`) was calculated manually using code like this:

```cpp
int n_lambda_with_tangential = 2 * c3_options_.num_friction_directions * c3_options_.num_contacts;
if (c3_options_.contact_model == "stewart_and_trinkle") {
    contact_model_ = c3::multibody::ContactModel::kStewartAndTrinkle;
    n_lambda_ = 2 * c3_options_.num_contacts + n_lambda_with_tangential;
} else if (c3_options_.contact_model == "anitescu") {
    contact_model_ = c3::multibody::ContactModel::kAnitescu;
    n_lambda_ = n_lambda_with_tangential;
} else {
    std::cerr << ("Unknown or unsupported contact model: " +
      c3_options_.contact_model) << std::endl;
    DRAKE_THROW_UNLESS(false);
}
```

This can now be replaced with the `GetNumContactVariables` utility function, which simplifies the calculation and ensures consistency across the codebase. The function automatically determines the number of contact variables based on the contact model, number of contacts, and friction directions.

#### Updated Code

```cpp
contact_model_ = c3::multibody::GetContactModelMap().at(c3_options_.contact_model);
n_lambda_ = c3::multibody::LCSFactory::GetNumContactVariables(
    contact_model_, c3_options_.num_contacts, c3_options_.num_friction_directions);
```

#### Benefits of Using `GetNumContactVariables`
1. **Readability**: The function abstracts away the manual calculations, making the code easier to read and maintain.
2. **Consistency**: Ensures that the calculation logic is centralized and consistent across the codebase.
3. **Error Handling**: Automatically validates the contact model and other parameters, reducing the likelihood of runtime errors.

#### Example Migration

**Old Code:**
```cpp
int n_lambda_with_tangential = 2 * c3_options_.num_friction_directions * c3_options_.num_contacts;
if (c3_options_.contact_model == "stewart_and_trinkle") {
    contact_model_ = c3::multibody::ContactModel::kStewartAndTrinkle;
    n_lambda_ = 2 * c3_options_.num_contacts + n_lambda_with_tangential;
} else if (c3_options_.contact_model == "anitescu") {
    contact_model_ = c3::multibody::ContactModel::kAnitescu;
    n_lambda_ = n_lambda_with_tangential;
} else {
    std::cerr << ("Unknown or unsupported contact model: " +
      c3_options_.contact_model) << std::endl;
    DRAKE_THROW_UNLESS(false);
}
```

**New Code:**
```cpp
contact_model_ = c3::multibody::GetContactModelMap().at(c3_options_.contact_model);
n_lambda_ = c3::multibody::LCSFactory::GetNumContactVariables(
    contact_model_, c3_options_.num_contacts, c3_options_.num_friction_directions);
```

By using `GetNumContactVariables`, you can simplify your code and reduce the risk of errors during manual calculations.

