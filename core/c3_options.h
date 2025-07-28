#pragma once

#include "drake/common/yaml/yaml_io.h"
#include "drake/common/yaml/yaml_read_archive.h"

namespace c3 {

struct C3Options {
  // Hyperparameters
  bool warm_start =
      true;  // Use results of current admm iteration as warm start for next
  bool end_on_qp_step =
      true;  // If false, Run a half step calculating the state using the LCS
  bool scale_lcs =
      true;  // Scale the LCS matrices by a factor of norm(A[0])/norm(D[0])

  int num_threads = 10;  // 0 is dynamic, greater than 0 for a fixed count
  int delta_option =
      1;  // 1 initializes the state value of the delta value with x0

  double M = 1000;  // big M value for MIQP

  int admm_iter = 3;  // total number of ADMM iterations

  // See comments below for how we parse the .yaml into the cost matrices
  double gamma;          // scaling factor for state and input the cost matrices
  float rho_scale = 10;  // scaling of rho parameter (/rho = rho_scale * /rho)
  Eigen::MatrixXd Q;
  Eigen::MatrixXd R;
  Eigen::MatrixXd G;
  Eigen::MatrixXd U;

  // Uniform scaling of the cost matrices.
  // Q = w_Q * diag(q_vector)
  // R = w_R * diag(r_vector)
  // G = w_G * diag(g_vector)
  // U = w_U * diag(u_vector)
  double w_Q;
  double w_R;
  double w_G;
  double w_U;

  // Unused except when parsing the costs from a yaml
  // We assume a diagonal Q, R, G, U matrix, so we can just specify the diagonal
  // terms as *_vector. To make indexing even easier, we split the parsing of
  // the g_vector and u_vector into the x, lambda, and u terms. *_lambda are the
  // default weights. If not specified, the Stewart and Trinkle formulation of
  // *_gamma, *_lambda_n, *_lambda_t will be used.
  std::vector<double> q_vector;
  std::vector<double> r_vector;

  std::vector<double> g_vector;
  std::vector<double> g_x;
  std::vector<double> g_gamma;
  std::vector<double> g_lambda_n;
  std::vector<double> g_lambda_t;
  std::vector<double> g_lambda;
  std::vector<double> g_u;

  std::vector<double> u_vector;
  std::vector<double> u_x;
  std::vector<double> u_gamma;
  std::vector<double> u_lambda_n;
  std::vector<double> u_lambda_t;
  std::vector<double> u_lambda;
  std::vector<double> u_u;

  template <typename Archive>
  void Serialize(Archive* a) {
    a->Visit(DRAKE_NVP(warm_start));
    a->Visit(DRAKE_NVP(end_on_qp_step));
    a->Visit(DRAKE_NVP(scale_lcs));

    a->Visit(DRAKE_NVP(num_threads));
    a->Visit(DRAKE_NVP(delta_option));

    a->Visit(DRAKE_NVP(M));

    a->Visit(DRAKE_NVP(admm_iter));

    a->Visit(DRAKE_NVP(rho_scale));
    a->Visit(DRAKE_NVP(gamma));

    a->Visit(DRAKE_NVP(w_Q));
    a->Visit(DRAKE_NVP(w_R));
    a->Visit(DRAKE_NVP(w_G));
    a->Visit(DRAKE_NVP(w_U));
    a->Visit(DRAKE_NVP(q_vector));
    a->Visit(DRAKE_NVP(r_vector));
    a->Visit(DRAKE_NVP(g_x));
    a->Visit(DRAKE_NVP(g_gamma));
    a->Visit(DRAKE_NVP(g_lambda_n));
    a->Visit(DRAKE_NVP(g_lambda_t));
    a->Visit(DRAKE_NVP(g_lambda));
    a->Visit(DRAKE_NVP(g_u));
    a->Visit(DRAKE_NVP(u_x));
    a->Visit(DRAKE_NVP(u_gamma));
    a->Visit(DRAKE_NVP(u_lambda_n));
    a->Visit(DRAKE_NVP(u_lambda_t));
    a->Visit(DRAKE_NVP(u_lambda));
    a->Visit(DRAKE_NVP(u_u));

    g_vector = std::vector<double>();
    g_vector.insert(g_vector.end(), g_x.begin(), g_x.end());
    if (g_lambda.empty()) {
      g_lambda.insert(g_lambda.end(), g_gamma.begin(), g_gamma.end());
      g_lambda.insert(g_lambda.end(), g_lambda_n.begin(), g_lambda_n.end());
      g_lambda.insert(g_lambda.end(), g_lambda_t.begin(), g_lambda_t.end());
    }
    g_vector.insert(g_vector.end(), g_lambda.begin(), g_lambda.end());

    g_vector.insert(g_vector.end(), g_u.begin(), g_u.end());
    u_vector = std::vector<double>();
    u_vector.insert(u_vector.end(), u_x.begin(), u_x.end());
    if (u_lambda.empty()) {
      u_lambda.insert(u_lambda.end(), u_gamma.begin(), u_gamma.end());
      u_lambda.insert(u_lambda.end(), u_lambda_n.begin(), u_lambda_n.end());
      u_lambda.insert(u_lambda.end(), u_lambda_t.begin(), u_lambda_t.end());
    }

    u_vector.insert(u_vector.end(), u_lambda.begin(), u_lambda.end());
    u_vector.insert(u_vector.end(), u_u.begin(), u_u.end());

    Eigen::VectorXd q = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(
        this->q_vector.data(), this->q_vector.size());
    Eigen::VectorXd r = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(
        this->r_vector.data(), this->r_vector.size());
    Eigen::VectorXd g = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(
        this->g_vector.data(), this->g_vector.size());
    Eigen::VectorXd u = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(
        this->u_vector.data(), this->u_vector.size());

    DRAKE_DEMAND(g.size() == u.size());

    Q = w_Q * q.asDiagonal();
    R = w_R * r.asDiagonal();
    G = w_G * g.asDiagonal();
    U = w_U * u.asDiagonal();
  }
};

inline C3Options LoadC3Options(const std::string& filename) {
  auto options = drake::yaml::LoadYamlFile<C3Options>(filename);
  return options;
}

}  // namespace c3