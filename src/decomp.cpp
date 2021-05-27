#include "decomp.h"

#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

Cx5 LowRank(Cx5 const &mIn, long const nRetain, Log const &log)
{
  long const rows = mIn.dimension(0) * mIn.dimension(1) * mIn.dimension(2) * mIn.dimension(3);
  Eigen::Map<Eigen::MatrixXcf const> m(mIn.data(), rows, mIn.dimension(4));
  Cx5 out(mIn.dimension(0), mIn.dimension(1), mIn.dimension(2), mIn.dimension(3), nRetain);
  Eigen::Map<Eigen::MatrixXcf> lr(out.data(), rows, nRetain);
  log.info(FMT_STRING("SVD dimensions {}x{}"), m.rows(), m.cols());
  auto const svd = m.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
  log.info(FMT_STRING("Retaining {} singular vectors for low-rank matrix"), nRetain);
  Eigen::VectorXf const vals = svd.singularValues();
  lr = svd.matrixU().leftCols(nRetain) * vals.head(nRetain).asDiagonal() *
       svd.matrixV().leftCols(nRetain).adjoint();
  return out;
}

Cx2 Covariance(Cx2 const &data)
{
  Cx const scale(1.f / data.dimension(1));
  return data.conjugate().contract(data, Eigen::IndexPairList<Eigen::type2indexpair<1, 1>>()) *
         scale;
}

void PCA(Cx2 const &gramIn, Cx2 &vecIn, R1 &valIn)
{
  Eigen::Map<Eigen::MatrixXcf const> gram(gramIn.data(), gramIn.dimension(0), gramIn.dimension(1));
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcf> es;
  es.compute(gram);
  Eigen::Map<Eigen::MatrixXcf> vec(vecIn.data(), vecIn.dimension(0), vecIn.dimension(1));
  Eigen::Map<Eigen::VectorXf> val(valIn.data(), valIn.dimension(0));
  assert(vec.rows() == gram.rows());
  assert(vec.cols() == gram.cols());
  assert(val.rows() == gram.rows());
  vec = es.eigenvectors();
  val = es.eigenvalues();
}