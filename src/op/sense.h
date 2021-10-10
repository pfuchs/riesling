#pragma once

#include "operator.h"

template <int R>
struct SenseOpR final : Operator<R, R + 1>
{
  using Input = typename Operator<R, R + 1>::Input;
  using Output = typename Operator<R, R + 1>::Output;

  SenseOpR(Cx4 &maps, typename Output::Dimensions const &fullSize);

  void A(Input const &x, Output &y) const;
  void Adj(Output const &x, Input &y) const;
  void AdjA(Input const &x, Input &y) const;

  long channels() const;
  Sz3 dimensions() const;

private:
  Output maps_;
  typename Output::Dimensions left_, right_;
};

using SenseOp = SenseOpR<3>;
using SenseBasis = SenseOpR<4>;
