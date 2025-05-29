import gin
import numpy as np
import gym
import os
import casadi as csd
from scipy import sparse
from scipy.linalg import block_diag
from scipy import linalg

from typing import Tuple, Optional
from pydrake.systems.primitives import (
    PassThrough,
    Multiplexer,
    ConstantVectorSource,
)
from pydrake.systems.framework import LeafSystem
from pydrake.multibody.meshcat import ContactVisualizer, ContactVisualizerParams
from pydrake.multibody.math import SpatialForce
from pydrake.multibody.plant import ExternallyAppliedSpatialForce_
from pydrake.common.value import Value
from pydrake.common.cpp_param import List
from common.drake_system import BaseDrakeSystem
from common.env_configs import CartPoleSoftWallsDrakeSystemConfigs
from common.drake_gym_wrapper import DrakeGymEnv


@gin.configurable
class CartPoleSoftWallsSystem(BaseDrakeSystem):
    def __init__(self, configs: CartPoleSoftWallsDrakeSystemConfigs):
        super().__init__(drake_sys_configs=configs)

        # add collision/visualization geoms to MultibodyPlant. If not, finalize the plant
        self.plant.Finalize()

        # establish symbolic cost functions for MPC
        self.path_cost_fn, self.final_cost_fn = self._create_symbolic_cost_fn()

        if configs.enable_contact_visualizer:
            ContactVisualizer.AddToBuilder(
                self.builder,
                self.plant,
                self.meshcat,
                ContactVisualizerParams(radius=0.005),
            )

        # build actuation block
        actuation = self.builder.AddSystem(Multiplexer([1, 1]))
        prismatic_actuation_force = self.builder.AddSystem(PassThrough(1))
        # Zero torque to the revolute joint --it is underactuated.
        revolute_actuation_torque = self.builder.AddSystem(ConstantVectorSource([0]))
        self.builder.Connect(
            revolute_actuation_torque.get_output_port(),
            actuation.get_input_port(1),
        )
        self.builder.Connect(
            prismatic_actuation_force.get_output_port(),
            actuation.get_input_port(0),
        )
        self.builder.Connect(actuation.get_output_port(), self.plant.get_actuation_input_port())

        # build block to simulate forces that soft walls exert on the pole
        softwall_forces = self.builder.AddSystem(self.soft_wall_reaction_forces())
        self.builder.Connect(
            softwall_forces.get_output_port(),
            self.plant.get_applied_spatial_force_input_port(),
        )
        self.builder.Connect(self.plant.get_state_output_port(), softwall_forces.get_input_port())

        # build reward block
        reward_system = self.builder.AddSystem(self.design_reward_system())
        self.builder.Connect(
            self.plant.get_state_output_port(),
            reward_system.get_input_port(0),
        )
        self.builder.Connect(
            prismatic_actuation_force.get_output_port(),
            reward_system.get_input_port(1),
        )

        # export some ports for later use
        self.builder.ExportInput(prismatic_actuation_force.get_input_port(), "actions")
        self.builder.ExportOutput(self.plant.get_state_output_port(), "observations")
        self.builder.ExportOutput(reward_system.get_output_port(), "reward")
        self.builder.ExportOutput(self.plant.get_contact_results_output_port(), "contact_force")

    def design_reward_system(self) -> LeafSystem:
        if self.is_diagram_built:
            raise RuntimeError("The diagram was built, so it is not possible to design a reward system")

        class RewardSystem(LeafSystem):
            def __init__(self, env_system: CartPoleSoftWallsSystem):
                super(RewardSystem, self).__init__()
                self.env_system = env_system
                self.DeclareVectorInputPort("cartpole_state", self.env_system.state_size)
                self.DeclareVectorInputPort("action", 1)
                self.DeclareVectorOutputPort("reward", 1, self.CalcReward)

            def CalcReward(self, context, output):
                cartpole_state = self.get_input_port(0).Eval(context)
                actions = self.get_input_port(1).Eval(context)

                # we basically evaluate the symbolic cost function with real numbers
                output[0] = 15 - self.env_system.path_cost_fn(cartpole_state, actions)

        return RewardSystem(self)

    def soft_wall_reaction_forces(self) -> LeafSystem:
        if self.is_diagram_built:
            raise RuntimeError("The diagram was built, so it is not possible to design a reward system")

        class SoftWallReactionForces(LeafSystem):
            def __init__(self, env_system: CartPoleSoftWallsSystem):
                super(SoftWallReactionForces, self).__init__()
                self.env_system = env_system
                forces_cls = Value[List[ExternallyAppliedSpatialForce_[float]]]
                self.DeclareAbstractOutputPort(
                    "spatial_forces",
                    lambda: forces_cls(),
                    self.CalcSoftWallForces,
                )
                self.DeclareVectorInputPort("cartpole_state", self.env_system.state_size)
                self.pole_body = self.env_system.plant.GetBodyByName("Pole")

            def CalcSoftWallForces(self, context, spatial_forces_vector):
                force = ExternallyAppliedSpatialForce_[float]()
                force.body_index = self.pole_body.index()
                force.p_BoBq_B = self.pole_body.default_com()

                # Calculate force exerts on the tip of the pole along x-axis given the current state
                cartpole_state = self.get_input_port(0).Eval(context)
                left_wall_xpos = -0.35  # ensure those values match sdf file
                right_wall_xpos = 0.35
                pole_length = 0.6
                pole_tip_xpos = cartpole_state[0] - pole_length * np.sin(cartpole_state[1])

                left_wall_force = max(0.0, left_wall_xpos - pole_tip_xpos) * self.env_system.sys_configs.wall_stiffness
                right_wall_force = (
                    min(0.0, right_wall_xpos - pole_tip_xpos) * self.env_system.sys_configs.wall_stiffness
                )

                if left_wall_force != 0:
                    wall_force = left_wall_force
                elif right_wall_force != 0:
                    wall_force = right_wall_force
                else:
                    wall_force = 0.0

                force.F_Bq_W = SpatialForce(tau=[0, 0, 0], f=[wall_force, 0, 0])
                spatial_forces_vector.set_value([force])

        return SoftWallReactionForces(self)

    def preprocess_action(self, action: np.array) -> np.array:
        """Convert action to the correct scale.

        To facilitate more stable RL, the action is often scaled to range [-1, 1]. Thus, before passing
        actions to system's actuators, they need to be converted to the right scale.
        """
        return action

    def _create_symbolic_cost_fn(
        self,
    ) -> Tuple[csd.Function, csd.Function]:
        """Establish symbolic expressions of path cost and final cost function using Casadi.

        Returns:
            Tuple[csd.Function, csd.Function]: the path and terminal cost function in symbolic form.
        """
        state = csd.SX.sym("state", self.state_size)
        action = csd.SX.sym("action", 1)

        path_cost = (
            csd.dot(state, self.sys_configs.weight_path_state_cost @ state)
            + self.sys_configs.w_action * action * action
        )
        final_cost = csd.dot(state, self.sys_configs.weight_terminal_state_cost @ state)

        # wrap cost functions by Function objects in Casadi
        path_cost_fn = csd.Function("path_cost_fn", [state, action], [path_cost])
        final_cost_fn = csd.Function("final_cost_fn", [state], [final_cost])

        return path_cost_fn, final_cost_fn

    def init_LCS_params(self):
        dt = self.sys_configs.frame_skip * self.sys_configs.sim_dt
        # using discrete-time linear model of true dynamics
        A = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 3.51, 0, 0], [0, 22.2, 0, 0]])
        A = A * dt + np.eye(4)
        A = A.ravel(order="F")
        # B = np.array([0, 0, 1.02, 1.7]).ravel(order="F") * dt
        B = np.array([0, 0, 1.42, 1.7]).ravel(order="F") * dt
        C = np.array([[0, 0], [0, 0], [0, 0], [4.7619, -4.7619]]).ravel(order="F") * dt
        d = np.array([0, 0, 0, 0]).ravel(order="F") * dt
        F = np.array([[0.0014, 0], [0, 0.0014]]).ravel(order="F")
        D = np.array([[-1.0, 0.6, 0, 0], [1, -0.6, 0, 0]]).ravel(order="F")
        D = np.array([[-1.0, 0.6, 0, 0], [1, -0.6, 0, 0]]).ravel(order="F")
        E = np.array([0, 0]).ravel(order="F")
        c = np.array([1.0, 1.0]).ravel(order="F")

        aux_guess = np.concatenate([A, B, C, d, D, E, F]).copy()
        # perturb_noise = np.random.uniform(low=0.01, high=0.1, size=aux_guess.shape)

        # return aux_guess + perturb_noise
        return aux_guess


def create_cartpole_softwalls_gym_env():
    """Return a gym-compatible class for cartpole with soft walls."""
    path_to_config_file = os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            "../../configs/drake_env_configs/cartpole_softwalls_drake_env_config.gin",
        )
    )
    gin.parse_config_file(path_to_config_file)

    drake_system = CartPoleSoftWallsSystem(configs=gin.REQUIRED)
    drake_system.build_diagram()

    action_space = gym.spaces.Box(
        low=np.array([-drake_system.sys_configs.action_threshold], dtype="float32"),
        high=np.array([drake_system.sys_configs.action_threshold], dtype="float32"),
    )

    observation_space = gym.spaces.Box(
        low=np.asarray(drake_system.sys_configs.min_obs, dtype="float32"),
        high=np.asarray(drake_system.sys_configs.max_obs, dtype="float32"),
    )

    get_initial_state_fn = (
        lambda: np.random.uniform(drake_system.sys_configs.min_rand, drake_system.sys_configs.max_rand)
        if drake_system.sys_configs.rand_init_state
        else drake_system.sys_configs.init_state
    )

    return DrakeGymEnv(
        simulator=drake_system.sim,
        time_step=drake_system.sys_configs.sim_dt * drake_system.sys_configs.frame_skip,
        high_level_action_space=action_space,
        high_level_observation_space=observation_space,
        reward="reward",
        preprocess_action_fn=drake_system.preprocess_action,
        action_port_id="actions",
        observation_port_id="observations",
        get_initial_state_fn=get_initial_state_fn,
        path_cost_fn=drake_system.path_cost_fn,
        final_cost_fn=drake_system.final_cost_fn,
        # init_LCS_params=drake_system.init_LCS_params(),
    )
