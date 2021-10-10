#include "../tensorOps.h"
#include "../threads.h"
#include "sense.h"

template <int R>
SenseOpR<R>::SenseOpR(Output &maps, Output::Dimensions const &bigSize)
    : maps_{std::move(maps)}
{
  auto const mapSz = maps_.dimensions();
  std::fill_n(left_.begin(), R - 3, 0L);
  std::fill_n(right_.begin(), R - 3, 0L);
  std::transform(
      bigSize.end() - 3, bigSize.end(), mapSz.end() - 3, left_.end() - 3, [](long big, long small) {
        return (big - small + 1) / 2;
      });
  std::transform(
      bigSize.begin(), bigSize.end(), mapSz.end() - 3, right_.end() - 3, [](long big, long small) {
        return (big - small) / 2;
      });
}

template <int R>
long SenseOpR<R>::channels() const
{
  return maps_.dimension(0);
}

template <int R>
Sz3 SenseOpR<R>::dimensions() const
{
  return Sz3{maps_.dimension(1), maps_.dimension(2), maps_.dimension(3)};
}

template <int R>
void SenseOpR<R>::A(Input const &x, Output &y) const
{
  assert(y.dimension(0) == maps_.dimension(0));
  for (auto ii = 0; ii < R - 3; ii++) {
    assert(x.dimension(ii) == y.dimension(ii + 1));
  }
  for (auto ii = 0; ii < 3; ii++) {
    assert(x.dimension(R - 3 + ii) == maps_.dimension(1 + ii));
    assert(
        y.dimension(R - 3 + ii) ==
        (maps_.dimension(1 + ii) + left_[R - 3 + ii] + right_[R - 3 + ii]));
  }

  Output::Dimensions res, brd;
  for (auto ii = 0; ii < R - 3; ii++) {
    res[ii] = 1;
  }
  for (auto ii = 0; ii < 3; ii++) {
    res[ii + R - 3] = x.dimension(ii);
  }
  brd[0] = maps_.dimension(0);
  for (auto ii = 1; ii < R; ii++) {
    brd[ii] = 1;
  }

  Eigen::array<std::pair<int, int>, 4> paddings;
  std::transform(
      left_.begin(), left_.end(), right_.begin(), paddings.begin(), [](long left, long right) {
        return std::make_pair(left, right);
      });

  y.device(Threads::GlobalDevice()) = (x.reshape(res).broadcast(brd) * maps_).pad(paddings);
}

template <int R>
void SenseOpR<R>::Adj(Output const &x, Input &y) const
{
  assert(x.dimension(0) == maps_.dimension(0));
  for (auto ii = 0; ii < R - 3; ii++) {
    assert(y.dimension(ii) == x.dimension(ii + 1));
  }
  for (auto ii = 0; ii < 3; ii++) {
    assert(y.dimension(R - 3 + ii) == maps_.dimension(1 + ii));
    assert(
        x.dimension(R - 3 + ii) ==
        (maps_.dimension(1 + ii) + left_[R - 3 + ii] + right_[R - 3 + ii]));
  }
  y.device(Threads::GlobalDevice()) = ConjugateSum(x.slice(left_, size_), maps_);
}

template <int R>
void SenseOpR<R>::AdjA(Input const &x, Input &y) const
{
  y.device(Threads::GlobalDevice()) = x;
}

template struct SenseOpR<3>;
template struct SenseOpR<4>;
