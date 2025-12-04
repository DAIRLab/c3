# C3 Multibody Module

This module provides tools for creating Linear Complementarity Systems (LCS) from Drake MultibodyPlant models, enabling the use of C3 for contact-rich manipulation and locomotion tasks.

## Overview

The multibody module bridges Drake's rigid body dynamics with C3's complementarity-based control framework. It linearizes contact dynamics around operating points and formulates them as LCS problems.

## Key Components

### LCSFactory

The `LCSFactory` class linearizes a Drake `MultibodyPlant` into an LCS representation. It supports multiple contact models:

- **Stewart and Trinkle**: Full friction cone with normal and tangential forces
- **Anitescu**: Linearized friction cone approximation
- **Frictionless Spring**: Simple normal-only contact with spring compliance

### Contact Models

#### Stewart and Trinkle Model
Uses explicit complementarity constraints for normal and tangential contact forces with a polyhedral friction cone approximation.

#### Anitescu Model  
Combines normal and tangential forces into a single set of variables, resulting in a more compact formulation.

#### Frictionless Spring Model
Models contacts as compliant springs without friction, useful for simple contact scenarios.

## LCS Factory Options Configuration

The `LCSFactoryOptions` struct controls how the multibody plant is linearized into an LCS. These options are typically specified in a YAML configuration file.

### Basic Structure

```yaml
contact_model: "stewart_and_trinkle"  # Contact model type
num_contacts: 3                        # Number of contact pairs
spring_stiffness: 0                    # Spring stiffness (for frictionless_spring model)
num_friction_directions: 1             # Friction cone approximation resolution
mu: [0.1, 0.1, 0.1]                   # Friction coefficients per contact
N: 10                                  # Prediction horizon length
dt: 0.01                               # Time step size (seconds)
```

### Configuration Fields

#### Required Fields

- **`contact_model`** (string): Contact dynamics formulation
  - Options: `"stewart_and_trinkle"`, `"anitescu"`, `"frictionless_spring"`
  - Default: None (must be specified)

- **`num_contacts`** (int): Total number of contact pairs in the system
  - Must match the actual number of contacts or contact pair configurations

- **`N`** (int): Number of time steps in the prediction horizon
  - Typical values: 5-20 depending on task and computational budget

- **`dt`** (double): Time step duration in seconds
  - Typical values: 0.01-0.05
  - Smaller values give better accuracy but increase computation

#### Contact-Specific Fields

- **`spring_stiffness`** (double): Spring constant for `frictionless_spring` model
  - Only used when `contact_model: "frictionless_spring"`
  - Units: N/m
  - Higher values = stiffer contact

- **`num_friction_directions`** (int, optional): Number of friction cone edges per contact
  - Used with `stewart_and_trinkle` or `anitescu` models
  - Higher values = better friction cone approximation, more variables
  - Typical values: 1-4
  - If specified, applies to all contacts uniformly

- **`num_friction_directions_per_contact`** (vector<int>, optional): Per-contact friction directions
  - Allows different friction cone resolutions for different contacts
  - Length must equal `num_contacts`
  - Example: `[1, 2, 1]` for 3 contacts with varying resolutions

- **`mu`** (vector<double>, optional): Friction coefficients
  - Length must equal `num_contacts`
  - Typical values: 0.1-1.0
  - Required unless using `contact_pair_configs`

#### Advanced Contact Configuration

- **`contact_pair_configs`** (vector<ContactPairConfig>, optional): Detailed contact specifications

When using `contact_pair_configs`, each entry specifies:

```yaml
contact_pair_configs:
  - body_A: "cube"                           # First body name
    body_B: "platform"                        # Second body name
    body_A_collision_geom_indices: [0]       # Collision geometry indices on body A
    body_B_collision_geom_indices: [0]       # Collision geometry indices on body B
    num_friction_directions: 1               # Friction cone edges for this pair
    mu: 0.1                                  # Friction coefficient for this pair
```

**ContactPairConfig Fields:**
- `body_A` (string): Name of first body in contact pair
- `body_B` (string): Name of second body in contact pair
- `body_A_collision_geom_indices` (vector<int>): Indices of collision geometries on body A
- `body_B_collision_geom_indices` (vector<int>): Indices of collision geometries on body B
- `num_friction_directions` (int): Friction cone approximation for this pair
- `mu` (float): Friction coefficient for this pair

### Configuration Examples

#### Example 1: Simple Frictionless Contact

```yaml
# Simple spring-based contact (e.g., cartpole with soft walls)
contact_model: "frictionless_spring"
num_contacts: 2
spring_stiffness: 100
N: 5
dt: 0.01
```

#### Example 2: Uniform Friction Contacts

```yaml
# Multiple contacts with same friction properties
contact_model: "stewart_and_trinkle"
num_contacts: 3
num_friction_directions: 1
mu: [0.1, 0.1, 0.1]
spring_stiffness: 0
N: 10
dt: 0.01
```

#### Example 3: Heterogeneous Contact Configuration

```yaml
# Cube manipulation with different contact properties
contact_model: "stewart_and_trinkle"
num_contacts: 3
spring_stiffness: 0
N: 10
dt: 0.01

contact_pair_configs:
  - body_A: "cube"
    body_A_collision_geom_indices: [0]
    body_B: "left_finger"
    body_B_collision_geom_indices: [0]
    mu: 0.1
    num_friction_directions: 1
    
  - body_A: "cube"
    body_A_collision_geom_indices: [0]
    body_B: "right_finger"
    body_B_collision_geom_indices: [0]
    mu: 0.1
    num_friction_directions: 1
    
  - body_A: "cube"
    body_A_collision_geom_indices: [0]
    body_B: "platform"
    body_B_collision_geom_indices: [0]
    mu: 0.5  # Higher friction with platform
    num_friction_directions: 2  # Finer friction cone approximation
```

#### Example 4: Anitescu Model

```yaml
# Compact friction formulation
contact_model: "anitescu"
num_contacts: 2
num_friction_directions: 2
mu: [0.2, 0.3]
spring_stiffness: 0
N: 8
dt: 0.02
```

### Loading Options from YAML

In C++:
```cpp
#include "multibody/lcs_factory_options.h"

// Load from file
auto options = c3::LoadLCSFactoryOptions("path/to/config.yaml");

// Use with LCSFactory
c3::multibody::LCSFactory factory(plant, context, plant_ad, context_ad, options);
factory.UpdateStateAndInput(state, input);
auto lcs = factory.GenerateLCS();
```

In Python:
```python
from pydrake.all import yaml_load_typed
from pyc3 import LCSFactoryOptions

# Create options from yaml
options = yaml_load_typed("path/to/config.yaml", schema=LCSFactoryOptions)
```

## See Also

- **Examples**: Check the examples directory for complete working examples with different contact models
- **C3 Options**: See c3_options.h for solver configuration