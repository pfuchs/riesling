#pragma once

#include "../fft_plan.h"
#include "operator.h"

template <int Rank, int FFTRank = 3>
struct FFTOp final : Operator<Rank, Rank>
{
  using Parent = Operator<Rank, Rank>;
  using Input = typename Parent::Input;
  using InputDims = typename Parent::InputDims;
  using Output = typename Parent::Output;
  using OutputDims = typename Parent::OutputDims;
  using Tensor = typename Eigen::Tensor<Cx, Rank>;

  FFTOp(InputDims const &dims)
    : dims_{dims}
    , fft_{dims_}
  {
  }

  InputDims inputDimensions() const
  {
    return dims_;
  }

  OutputDims outputDimensions() const
  {
    return dims_;
  }

  // Need these for Eigen expressions
  template <typename T>
  Tensor A(T const &x) const
  {
    Log::Debug("Out-of-place forward FFT op");
    Tensor y = x;
    fft_.forward(y);
    return y;
  }

  template <typename T>
  Tensor Adj(T const &x) const
  {
    Log::Debug("Out-of-place forward FFT op");
    Tensor y = x;
    fft_.reverse(y);
    return y;
  }

  // Provide these for in-place when we can
  Tensor &A(Tensor &x) const
  {
    Log::Debug("In-place forward FFT op");
    fft_.forward(x);
    return x;
  }

  Tensor &Adj(Tensor &x) const
  {
    Log::Debug("In-place adjoint FFT op");
    fft_.reverse(x);
    return x;
  }

private:
  InputDims dims_;
  FFT::Planned<5, 3> fft_;
};
