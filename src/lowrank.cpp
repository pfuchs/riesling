#include "lowrank.h"

#include <Eigen/SVD>

Cx5 LowRank(Cx5 const &mIn, long const nRetain)
{
  long const rows = mIn.dimension(0) * mIn.dimension(1) * mIn.dimension(2) * mIn.dimension(3);
  Eigen::Map<Eigen::MatrixXcf> m(mIn.data(), rows, mIn.dimension(4));
  Cx5 out(mIn.dimension(0), mIn.dimension(1), mIn.dimension(2), mIn.dimension(3), nRetain);
  Eigen::Map<Eigen::MatrixXcf> lr(out.data(), rows, nRetain);
  auto const svd = m.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
  Eigen::VectorXf const vals = svd.singularValues();
  lr = svd.matrixU().leftCols(nRetain) * vals.head(nRetain).asDiagonal() *
       svd.matrixV().leftCols(nRetain).adjoint();
  return out;
}