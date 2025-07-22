#include <gtest/gtest.h>

#include "systems/framework/c3_output.h"
#include "systems/framework/timestamped_vector.h"

namespace c3 {
namespace systems {
namespace test {

// Test fixture for framework-related tests.
class FrameworkTest : public ::testing::Test {
 protected:
    void SetUp() override {}
};

// Tests construction and data access for TimestampedVector.
TEST_F(FrameworkTest, TimestampedVectorConstructor) {
    // Construct from initializer list.
    TimestampedVector<double> vec{1.0, 2.0, 3.0};
    EXPECT_EQ(vec.data_size(), 3);
    Eigen::VectorXd expected(3);
    expected << 1.0, 2.0, 3.0;
    EXPECT_TRUE(vec.get_data().isApprox(expected));

    // Construct from Eigen vector.
    TimestampedVector<double> vec2(expected);
    EXPECT_TRUE(vec2.get_data().isApprox(expected));
}

// Tests data and timestamp accessors/mutators for TimestampedVector.
TEST_F(FrameworkTest, TimestampedVectorAccess) {
    // Construct with size.
    TimestampedVector<double> vec(3);
    EXPECT_EQ(vec.data_size(), 3);

    // Set and get data.
    Eigen::VectorXd data(3);
    data << 1.0, 2.0, 3.0;
    vec.SetDataVector(data);
    EXPECT_TRUE(vec.get_data().isApprox(data));

    // Set and get timestamp.
    vec.set_timestamp(42.0);
    EXPECT_DOUBLE_EQ(vec.get_timestamp(), 42.0);

    // Copy data vector without timestamp.
    Eigen::VectorXd no_ts = vec.CopyVectorNoTimestamp();
    EXPECT_TRUE(no_ts.isApprox(data));

    // Clone and verify deep copy.
    auto clone = vec.Clone();
    EXPECT_TRUE(clone->get_data().isApprox(data));
    EXPECT_DOUBLE_EQ(clone->get_timestamp(), 42.0);
}

// Tests construction and shape of C3Solution and C3Intermediates.
TEST_F(FrameworkTest, C3SolutionAndIntermediatesConstruction) {
    int n_x = 2, n_lambda = 3, n_u = 1, N = 4;
    C3Output::C3Solution sol(n_x, n_lambda, n_u, N);
    C3Output::C3Intermediates interm(n_x, n_lambda, n_u, N);

    // Verify solution matrix shapes.
    EXPECT_EQ(sol.x_sol_.rows(), n_x);
    EXPECT_EQ(sol.x_sol_.cols(), N);
    EXPECT_EQ(sol.lambda_sol_.rows(), n_lambda);
    EXPECT_EQ(sol.lambda_sol_.cols(), N);
    EXPECT_EQ(sol.u_sol_.rows(), n_u);
    EXPECT_EQ(sol.u_sol_.cols(), N);
    EXPECT_EQ(sol.time_vector_.size(), N);

    // Verify intermediates matrix shapes.
    EXPECT_EQ(interm.z_.rows(), n_x + n_lambda + n_u);
    EXPECT_EQ(interm.z_.cols(), N);
    EXPECT_EQ(interm.delta_.rows(), n_x + n_lambda + n_u);
    EXPECT_EQ(interm.delta_.cols(), N);
    EXPECT_EQ(interm.w_.rows(), n_x + n_lambda + n_u);
    EXPECT_EQ(interm.w_.cols(), N);
    EXPECT_EQ(interm.time_vector_.size(), N);

    // Construct C3Output from solution and intermediates.
    C3Output output(sol, interm);
}

}  // namespace test
}  // namespace systems
}  // namespace c3
