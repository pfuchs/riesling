#pragma once

#include "../kernel.h"
#include "../trajectory.h"
#include "operator.h"

template <int InPlane, int ThroughPlane>
struct GridKB final : Operator<4, 3>
{
  GridKB(Mapping map, bool const unsafe, Log &log);

  void A(Input const &x, Output &y) const;
  void Adj(Output const &x, Input &y) const;

  Input::Dimensions inSize() const;
  Output::Dimensions outSize() const;

  Sz3 gridDims() const;                        //!< Returns the dimensions of the grid
  Cx4 newMultichannel(long const nChan) const; //!< Returns a correctly sized multi-channel grid
  void setSDCExponent(float const dce);
  void setSDC(float const dc);
  void setSDC(R2 const &sdc);
  void setUnsafe();
  void setSafe();

  using InPlaneArray = Eigen::TensorFixedSize<float, Eigen::Sizes<InPlane>>;
  using ThroughPlaneArray = Eigen::TensorFixedSize<float, Eigen::Sizes<ThroughPlane>>;
  using Kernel = Eigen::TensorFixedSize<float, Eigen::Sizes<InPlane, InPlane, ThroughPlane>>;

private:
  Mapping mapping_;
  bool safe_;
  Log &log_;
  float DCexp_, betaIn_, betaThrough_;
  InPlaneArray indIn_;
  ThroughPlaneArray indThrough_;
  void kernel(Point3 const offset, float const dc, Kernel &k) const;
};

using GridKB3D = GridKB<3, 3>;
using GridKB2D = GridKB<3, 1>;
