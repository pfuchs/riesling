#pragma once

#include "../kernel.h"
#include "../trajectory.h"
#include "operator.h"

struct GridNN final : Operator<4, 3>
{
  GridNN(Mapping map, bool const unsafe, Log &log);

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

private:
  Mapping mapping_;
  bool safe_;
  Log &log_;
  float DCexp_;
};
