#include "eig.hpp"

namespace rl {

auto PowerMethod(std::shared_ptr<Ops::Op<Cx>> A, Index const iterLimit) -> PowerReturn
{
  Log::Print<Log::Level::High>("Power Method for A'A");
  Eigen::VectorXcf vec = Eigen::VectorXcf::Random(A->cols());
  float            val = vec.norm();
  vec /= val;
  for (auto ii = 0; ii < iterLimit; ii++) {
    vec = A->adjoint(A->forward(vec));
    val = vec.norm();
    vec /= val;
    Log::Print<Log::Level::High>("Iteration {} Eigenvalue {}", ii, val);
  }

  return {val, vec};
}

auto PowerMethodForward(std::shared_ptr<Ops::Op<Cx>> A, std::shared_ptr<Ops::Op<Cx>> P, Index const iterLimit) -> PowerReturn
{
  Log::Print<Log::Level::High>("Power Method for A'PA");
  Eigen::VectorXcf vec = Eigen::VectorXcf::Random(A->cols());
  float            val = vec.norm();
  vec /= val;
  for (auto ii = 0; ii < iterLimit; ii++) {
    vec = A->adjoint(P->adjoint(A->forward(vec)));
    val = vec.norm();
    vec /= val;
    Log::Print<Log::Level::High>("Iteration {} Eigenvalue {}", ii, val);
  }

  return {val, vec};
}

auto PowerMethodAdjoint(std::shared_ptr<Ops::Op<Cx>> A, std::shared_ptr<Ops::Op<Cx>> P, Index const iterLimit) -> PowerReturn
{
  Log::Print<Log::Level::High>("Power Method for adjoint system PAA'");
  Eigen::VectorXcf vec = Eigen::VectorXcf::Random(A->rows());
  float            val = vec.norm();
  vec /= val;
  for (auto ii = 0; ii < iterLimit; ii++) {
    vec = P->adjoint(A->forward(A->adjoint(vec)));
    val = vec.norm();
    vec /= val;
    Log::Print<Log::Level::High>("Iteration {} Eigenvalue {}", ii, val);
  }

  return {val, vec};
}

} // namespace rl
