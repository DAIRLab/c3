"""Smoke tests for c3 trajectory evaluation bindings.

These tests are primarily smoke tests designed to verify that the Python
bindings are working correctly and can be called without errors. They are
not comprehensive functional tests of the underlying C++ trajectory evaluation
algorithms. The tests use simple synthetic data to ensure basic binding
functionality rather than validating mathematical correctness or performance.
"""

import unittest
import numpy as np
import c3
import traj_eval


class TestTrajectoryEvaluator(unittest.TestCase):
    def setUp(self):
        # Create simple test data following the patterns from core_test.cc
        self.n_x, self.n_u, self.n_lambda = 4, 2, 2
        self.N = 3
        self.dt = 0.01

        # Create simple LCS similar to core_test.cc
        self.lcs = c3.LCS(
            np.eye(self.n_x),
            np.ones((self.n_x, self.n_u)),
            np.ones((self.n_x, self.n_lambda)),
            np.zeros(self.n_x),
            np.ones((self.n_lambda, self.n_x)),
            np.eye(self.n_lambda),
            np.ones((self.n_lambda, self.n_u)),
            np.ones(self.n_lambda),
            self.N,
            self.dt,
        )

        # Create test trajectories with non-zero values to get positive cost
        self.x_traj = [np.ones(self.n_x) for _ in range(self.N + 1)]
        self.u_traj = [np.ones(self.n_u) for _ in range(self.N)]
        self.x_des = [np.zeros(self.n_x) for _ in range(self.N + 1)]

        # Create cost matrices - ensure they are proper numpy arrays with correct dtype
        self.Q_matrix = np.eye(self.n_x, dtype=np.float64)
        self.R_matrix = np.eye(self.n_u, dtype=np.float64)
        # For list of matrices, ensure each is a separate array
        self.Q_matrices = [
            np.eye(self.n_x, dtype=np.float64) for _ in range(self.N + 1)
        ]
        self.R_matrices = [np.eye(self.n_u, dtype=np.float64) for _ in range(self.N)]

    def test_compute_quadratic_trajectory_cost_basic(self):
        # Test with single matrices
        cost = traj_eval.TrajectoryEvaluator.ComputeQuadraticTrajectoryCost(
            self.x_traj, self.x_des, self.Q_matrix
        )
        self.assertGreater(cost, 0)

        # Test with matrix lists
        cost = traj_eval.TrajectoryEvaluator.ComputeQuadraticTrajectoryCost(
            self.x_traj, self.x_des, self.Q_matrices
        )
        self.assertGreater(cost, 0)

    def test_compute_quadratic_trajectory_cost_single_desired(self):
        # Test with single desired vector
        x_des_single = np.zeros(self.n_x)
        cost = traj_eval.TrajectoryEvaluator.ComputeQuadraticTrajectoryCost(
            self.x_traj, x_des_single, self.Q_matrix
        )
        self.assertGreater(cost, 0)

        # Test with single desired vector and matrix lists
        cost = traj_eval.TrajectoryEvaluator.ComputeQuadraticTrajectoryCost(
            self.x_traj, x_des_single, self.Q_matrices
        )
        self.assertGreater(cost, 0)

    def test_compute_quadratic_trajectory_cost_zero_desired(self):
        # Test assuming zero desired
        cost = traj_eval.TrajectoryEvaluator.ComputeQuadraticTrajectoryCost(
            self.x_traj, self.Q_matrix
        )
        self.assertGreater(cost, 0)

        # Test with matrix list
        cost = traj_eval.TrajectoryEvaluator.ComputeQuadraticTrajectoryCost(
            self.x_traj, self.Q_matrices
        )
        self.assertGreater(cost, 0)

    def test_compute_quadratic_trajectory_cost_with_c3(self):
        # Create a C3 solver similar to core_test.cc but with penalize_input_change = False
        opts = c3.C3Options()
        opts.Q = self.Q_matrix
        opts.R = self.R_matrix
        opts.G = np.eye(self.n_x + self.n_u + self.n_lambda)
        opts.U = np.eye(self.n_x + self.n_u + self.n_lambda)
        opts.warm_start = False
        opts.scale_lcs = False
        opts.end_on_qp_step = True
        opts.num_threads = 1
        opts.admm_iter = 1
        opts.M = 100.0
        opts.gamma = 1.0
        opts.rho_scale = 1.0
        opts.penalize_input_change = False  # This should prevent the constraint failure

        costs = c3.C3.CreateCostMatricesFromC3Options(opts, self.N)
        solver = c3.C3QP(self.lcs, costs, self.x_des, opts)
        solver.Solve(np.zeros(self.n_x))

        # Test cost computation from C3 solver
        cost = traj_eval.TrajectoryEvaluator.ComputeQuadraticTrajectoryCost(solver)
        self.assertGreaterEqual(cost, 0)

        # Test with custom matrices
        cost = traj_eval.TrajectoryEvaluator.ComputeQuadraticTrajectoryCost(
            solver, self.Q_matrices, self.R_matrices
        )
        self.assertGreaterEqual(cost, 0)

    def test_simulate_pd_control_with_lcs(self):
        # Test PD control simulation following the core_test.cc pattern
        # Kp and Kd should be n_x length with exactly n_u non-zero entries
        Kp = np.zeros(self.n_x)
        Kd = np.zeros(self.n_x)
        # Set first n_u entries to be non-zero (positions for Kp, velocities for Kd)
        Kp[0] = 10.0  # position control
        Kp[1] = 10.0  # position control
        Kd[2] = 1.0  # velocity damping
        Kd[3] = 1.0  # velocity damping

        x_sim, u_sim = traj_eval.TrajectoryEvaluator.SimulatePDControlWithLCS(
            self.x_traj, self.u_traj, Kp, Kd, self.lcs
        )

        self.assertEqual(len(x_sim), self.N + 1)
        self.assertEqual(len(u_sim), self.N)
        for x in x_sim:
            self.assertEqual(len(x), self.n_x)
        for u in u_sim:
            self.assertEqual(len(u), self.n_u)

    def test_simulate_pd_control_with_coarse_fine_lcs(self):
        # Create fine LCS with smaller dt, following core_test.cc pattern
        fine_lcs = c3.LCS(
            np.eye(self.n_x),
            np.ones((self.n_x, self.n_u)) / 2.0,  # Scale B matrix for finer time step
            np.ones((self.n_x, self.n_lambda)),
            np.zeros(self.n_x),
            np.ones((self.n_lambda, self.n_x)),
            np.eye(self.n_lambda),
            np.ones((self.n_lambda, self.n_u)),
            np.ones(self.n_lambda),
            self.N * 2,  # Double the time steps
            self.dt / 2,  # Half the dt
        )

        # Set up Kp and Kd correctly as in the previous test
        Kp = np.zeros(self.n_x)
        Kd = np.zeros(self.n_x)
        Kp[0] = 10.0
        Kp[1] = 10.0
        Kd[2] = 1.0
        Kd[3] = 1.0

        x_sim, u_sim = traj_eval.TrajectoryEvaluator.SimulatePDControlWithLCS(
            self.x_traj, self.u_traj, Kp, Kd, self.lcs, fine_lcs
        )

        self.assertEqual(len(x_sim), self.N + 1)
        self.assertEqual(len(u_sim), self.N)

    def test_simulate_lcs_over_trajectory(self):
        config = c3.LCSSimulateConfig()

        # Test with initial state
        x_init = np.zeros(self.n_x)
        x_sim = traj_eval.TrajectoryEvaluator.SimulateLCSOverTrajectory(
            x_init, self.u_traj, self.lcs, config
        )

        self.assertEqual(len(x_sim), self.N + 1)
        for x in x_sim:
            self.assertEqual(len(x), self.n_x)

        # Test with trajectory plan
        x_sim = traj_eval.TrajectoryEvaluator.SimulateLCSOverTrajectory(
            self.x_traj, self.u_traj, self.lcs, config
        )

        self.assertEqual(len(x_sim), self.N + 1)

    def test_simulate_lcs_with_coarse_fine(self):
        # Create fine LCS
        fine_lcs = c3.LCS(
            np.eye(self.n_x),
            np.ones((self.n_x, self.n_u)),
            np.ones((self.n_x, self.n_lambda)),
            np.zeros(self.n_x),
            np.ones((self.n_lambda, self.n_x)),
            np.eye(self.n_lambda),
            np.ones((self.n_lambda, self.n_u)),
            np.ones(self.n_lambda),
            self.N * 2,
            self.dt / 2,
        )

        config = c3.LCSSimulateConfig()
        x_init = np.zeros(self.n_x)

        # Test with initial state
        x_sim = traj_eval.TrajectoryEvaluator.SimulateLCSOverTrajectory(
            x_init, self.u_traj, self.lcs, fine_lcs, config
        )

        self.assertEqual(len(x_sim), self.N + 1)

        # Test with trajectory
        x_sim = traj_eval.TrajectoryEvaluator.SimulateLCSOverTrajectory(
            self.x_traj, self.u_traj, self.lcs, fine_lcs, config
        )

        self.assertEqual(len(x_sim), self.N + 1)

    def test_zero_order_hold_trajectory(self):
        upsample_rate = 2

        # Test single trajectory
        upsampled = traj_eval.TrajectoryEvaluator.ZeroOrderHoldTrajectory(
            self.u_traj, upsample_rate
        )

        self.assertEqual(len(upsampled), len(self.u_traj) * upsample_rate)

        # Test state and input trajectories
        x_up, u_up = traj_eval.TrajectoryEvaluator.ZeroOrderHoldTrajectories(
            self.x_traj, self.u_traj, upsample_rate
        )

        # Following the pattern from core_test.cc:
        # x_traj has N+1 elements, u_traj has N elements
        # ZeroOrderHoldTrajectories upsamples x_traj[:-1] and adds back the last element
        self.assertEqual(len(x_up), (len(self.x_traj) - 1) * upsample_rate + 1)
        self.assertEqual(len(u_up), len(self.u_traj) * upsample_rate)

    def test_downsample_trajectory(self):
        # Create upsampled trajectory first
        upsample_rate = 2
        upsampled_u = traj_eval.TrajectoryEvaluator.ZeroOrderHoldTrajectory(
            self.u_traj, upsample_rate
        )

        # Downsample back
        downsampled = traj_eval.TrajectoryEvaluator.DownsampleTrajectory(
            upsampled_u, upsample_rate
        )

        self.assertEqual(len(downsampled), len(self.u_traj))

        # Test with trajectories
        x_up, u_up = traj_eval.TrajectoryEvaluator.ZeroOrderHoldTrajectories(
            self.x_traj, self.u_traj, upsample_rate
        )

        x_down, u_down = traj_eval.TrajectoryEvaluator.DownsampleTrajectories(
            x_up, u_up, upsample_rate
        )

        self.assertEqual(len(x_down), len(self.x_traj))
        self.assertEqual(len(u_down), len(self.u_traj))

    def test_check_coarse_and_fine_lcs_compatibility(self):
        # Create compatible fine LCS
        fine_lcs = c3.LCS(
            np.eye(self.n_x),
            np.ones((self.n_x, self.n_u)),
            np.ones((self.n_x, self.n_lambda)),
            np.zeros(self.n_x),
            np.ones((self.n_lambda, self.n_x)),
            np.eye(self.n_lambda),
            np.ones((self.n_lambda, self.n_u)),
            np.ones(self.n_lambda),
            self.N * 2,  # Double time steps
            self.dt / 2,  # Half dt
        )

        upsample_rate = (
            traj_eval.TrajectoryEvaluator.CheckCoarseAndFineLCSCompatibility(
                self.lcs, fine_lcs
            )
        )

        self.assertEqual(upsample_rate, 2)

    def test_check_lcs_and_trajectory_compatibility(self):
        # Test with valid trajectories - should not raise
        traj_eval.TrajectoryEvaluator.CheckLCSAndTrajectoryCompatibility(
            self.lcs, self.x_traj
        )

        # Test with state and input
        traj_eval.TrajectoryEvaluator.CheckLCSAndTrajectoryCompatibility(
            self.lcs, self.x_traj, self.u_traj
        )

        # Test with lambda trajectory - lambda should have N elements (not N+1)
        lambda_traj = [np.ones(self.n_lambda) for _ in range(self.N)]
        traj_eval.TrajectoryEvaluator.CheckLCSAndTrajectoryCompatibility(
            self.lcs, self.x_traj, self.u_traj, lambda_traj
        )

    def test_trajectory_compatibility_errors(self):
        # Test with wrong dimensions - should raise
        wrong_x = [np.ones(self.n_x + 1) for _ in range(self.N + 1)]

        with self.assertRaises(Exception):
            traj_eval.TrajectoryEvaluator.CheckLCSAndTrajectoryCompatibility(
                self.lcs, wrong_x
            )

        # Test with wrong length
        short_x = [np.ones(self.n_x) for _ in range(self.N)]  # Missing one state

        with self.assertRaises(Exception):
            traj_eval.TrajectoryEvaluator.CheckLCSAndTrajectoryCompatibility(
                self.lcs, short_x
            )


if __name__ == "__main__":
    unittest.main()
