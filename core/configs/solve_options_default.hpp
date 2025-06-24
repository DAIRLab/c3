#pragma once
#include <string>
const std::string default_solver_options =
    "print_to_console: 0\nlog_file_name: \"\"\nint_options:\n  max_iter: "
    "1000\n  # linsys_solver: 0\n  verbose: 0\n  warm_start: 1\n  polish: 1\n  "
    "scaled_termination: 1\n  check_termination: 25\n  polish_refine_iter: 3\n "
    " scaling: 10\n  adaptive_rho: 1\n  adaptive_rho_interval: "
    "0\n\ndouble_options:\n  rho: 0.0001\n  sigma: 1e-6\n  eps_abs: 1e-5\n  "
    "eps_rel: 1e-5\n  eps_prim_inf: 1e-5\n  eps_dual_inf: 1e-5\n  alpha: 1.6\n "
    " delta: 1e-6\n  adaptive_rho_tolerance: 5\n  adaptive_rho_fraction: 0.4\n "
    " time_limit: 1.0\n\nstring_options: {}";
