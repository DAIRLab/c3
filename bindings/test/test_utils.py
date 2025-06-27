from pyc3 import (
    TimestampedVector,
    C3Solution,
)

from pydrake.all import (
    LeafSystem,
    Value,
)


class C3Solution2Input(LeafSystem):
    def __init__(self, n_u):
        LeafSystem.__init__(self)
        self.n_u_ = n_u
        # Declare input port for C3 solutions.
        self.c3_solution_port_index_ = self.DeclareAbstractInputPort(
            "c3_solution", Value(C3Solution())
        ).get_index()
        # Declare output port for inputs.
        self.c3_input_port_index_ = self.DeclareVectorOutputPort(
            "u", n_u, self.GetC3Input
        ).get_index()

    # Getter for the input port.
    def get_input_port_c3_solution(self):
        return self.get_input_port(self.c3_solution_port_index_)

    # Getter for the output port.
    def get_output_port_c3_input(self):
        return self.get_output_port(self.c3_input_port_index_)

    def GetC3Input(self, context, output):
        input_value = self.EvalAbstractInput(context, 0)
        assert input_value is not None
        sol = input_value.get_value()
        output.SetFromVector(sol.u_sol[:, 0])


class Vector2TimestampedVector(LeafSystem):
    def __init__(self, n):
        LeafSystem.__init__(self)
        self.vector_port_index_ = self.DeclareVectorInputPort("state", n).get_index()
        self.timestamped_vector_port_index_ = self.DeclareVectorOutputPort(
            "timestamped_state", TimestampedVector(n), self.Convert
        ).get_index()

    # Getter for the input port.
    def get_input_port_state(self):
        return self.get_input_port(self.vector_port_index_)

    # Getter for the output port.
    def get_output_port_timestamped_state(self):
        return self.get_output_port(self.timestamped_vector_port_index_)

    def Convert(self, context, output):
        output.SetDataVector(self.EvalVectorInput(context, self.vector_port_index_).get_value())
        output.set_timestamp(context.get_time())  # Set timestamp.
