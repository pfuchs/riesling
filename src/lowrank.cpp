#include "lowrank.h"

#include <Eigen/SVD>

Eigen::MatrixXcf LowRank(Eigen::Map<Eigen::MatrixXcf> const &m, long const nRetain)
{
  auto const svd = m.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
  Eigen::VectorXf const vals = svd.singularValues();
  Eigen::MatrixXcf const lr = svd.matrixU().leftCols(nRetain) * vals.head(nRetain).asDiagonal() *
                              svd.matrixV().adjoint().leftCols(nRetain);
  return lr;
}