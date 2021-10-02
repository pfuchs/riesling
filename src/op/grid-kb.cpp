#include "grid-kb.h"

#include "../tensorOps.h"
#include "../threads.h"

#include <algorithm>
#include <cfenv>
#include <cmath>

template <int InPlane, int ThroughPlane>
GridKB<InPlane, ThroughPlane>::GridKB(Mapping map, bool const unsafe, Log &log)
    : mapping_{std::move(map)}
    , safe_{!unsafe}
    , log_{log}
    , DCexp_{1.f}
    , betaIn_{(float)M_PI * sqrtf(pow(InPlane * (mapping_.osamp - 0.5f) / mapping_.osamp, 2.f) - 0.8f)}
    , betaThrough_{
          (float)M_PI *
          sqrtf(pow(ThroughPlane * (mapping_.osamp - 0.5f) / mapping_.osamp, 2.f) - 0.8f)}
{

  // Array of indices used when building the kernel
  std::iota(indIn_.data(), indIn_.data() + InPlane, -InPlane / 2);
  std::iota(indThrough_.data(), indThrough_.data() + ThroughPlane, -ThroughPlane / 2);
}

template <int InPlane, int ThroughPlane>
Sz3 GridKB<InPlane, ThroughPlane>::gridDims() const
{
  return mapping_.cartDims;
}

template <int InPlane, int ThroughPlane>
Cx4 GridKB<InPlane, ThroughPlane>::newMultichannel(long const nc) const
{
  Cx4 g(nc, mapping_.cartDims[0], mapping_.cartDims[1], mapping_.cartDims[2]);
  g.setZero();
  return g;
}

template <int InPlane, int ThroughPlane>
void GridKB<InPlane, ThroughPlane>::setSDC(float const d)
{
  std::fill(mapping_.sdc.begin(), mapping_.sdc.end(), d);
}

template <int InPlane, int ThroughPlane>
void GridKB<InPlane, ThroughPlane>::setSDC(R2 const &sdc)
{
  std::transform(
      mapping_.noncart.begin(),
      mapping_.noncart.end(),
      mapping_.sdc.begin(),
      [&sdc](NoncartesianIndex const &nc) { return sdc(nc.read, nc.spoke); });
}

template <int InPlane, int ThroughPlane>
void GridKB<InPlane, ThroughPlane>::setSDCExponent(float const dce)
{
  DCexp_ = dce;
}

template <int InPlane, int ThroughPlane>
void GridKB<InPlane, ThroughPlane>::setUnsafe()
{
  safe_ = true;
}

template <int InPlane, int ThroughPlane>
void GridKB<InPlane, ThroughPlane>::setSafe()
{
  safe_ = false;
}

template <int W, typename T>
inline decltype(auto) KB(T const &x, float const beta)
{
  return (x > (W / 2.f))
      .select(
          x.constant(0.f),
          (x.constant(beta) * (x.constant(1.f) - (x * x.constant(2.f / W)).square()).sqrt())
              .bessel_i0());
}

template <int InPlane, int ThroughPlane>
void GridKB<InPlane, ThroughPlane>::kernel(Point3 const r, float const dc, Kernel &k) const
{
  InPlaneArray const kx = KB<InPlane>(indIn_.constant(r[0]) - indIn_, betaIn_);
  InPlaneArray const ky = KB<InPlane>(indIn_.constant(r[2]) - indIn_, betaIn_);

  if constexpr (ThroughPlane > 1) {
    ThroughPlaneArray const kz =
        KB<ThroughPlane>(indThrough_.constant(r[3]) - indThrough_, betaThrough_);
    k = Outer(Outer(kx, ky), kz);
  } else {
    k = Outer(kx, ky);
  }

  //   if (fft_) {
  //     Cx3 temp(sz_);
  //     temp = k.cast<Cx>();
  //     fft_->reverse(temp);
  //     temp.sqrt();
  //     fft_->forward(temp);
  //     k = temp.real();
  //   }
  k = k * dc / Sum(k);
}

template <int InPlane, int ThroughPlane>
void GridKB<InPlane, ThroughPlane>::Adj(Cx3 const &noncart, Cx4 &cart) const
{
  assert(noncart.dimension(0) == cart.dimension(0));
  assert(cart.dimension(1) == mapping_.cartDims[0]);
  assert(cart.dimension(2) == mapping_.cartDims[1]);
  assert(cart.dimension(3) == mapping_.cartDims[2]);
  assert(mapping_.sortedIndices.size() == mapping_.cart.size());

  long const nchan = cart.dimension(0);
  using FixZero = Eigen::type2index<0>;
  using FixOne = Eigen::type2index<1>;
  using FixIn = Eigen::type2index<InPlane>;
  using FixThrough = Eigen::type2index<ThroughPlane>;
  Eigen::IndexList<int, FixOne, FixOne, FixOne> rshNC;
  Eigen::IndexList<FixOne, FixIn, FixIn, FixThrough> brdNC;
  rshNC.set(0, nchan);

  Eigen::IndexList<FixOne, FixIn, FixIn, FixThrough> rshC;
  Eigen::IndexList<int, FixOne, FixOne, FixOne> brdC;
  brdC.set(0, nchan);

  Eigen::IndexList<int, FixIn, FixIn, FixThrough> szC;
  szC.set(0, nchan);

  auto dev = Threads::GlobalDevice();
  long const nThreads = dev.numThreads();
  std::vector<Cx4> workspace(nThreads);
  std::vector<long> minZ(nThreads, 0L), szZ(nThreads, 0L);
  auto grid_task = [&](long const lo, long const hi, long const ti) {
    // Allocate working space for this thread
    Kernel k;
    Eigen::IndexList<FixZero, int, int, int> stC;
    minZ[ti] = mapping_.cart[mapping_.sortedIndices[lo]].z - ((ThroughPlane - 1) / 2);

    if (safe_) {
      long const maxZ = mapping_.cart[mapping_.sortedIndices[hi - 1]].z + (ThroughPlane / 2);
      szZ[ti] = maxZ - minZ[ti] + 1;
      workspace[ti].resize(cart.dimension(0), cart.dimension(1), cart.dimension(2), szZ[ti]);
      workspace[ti].setZero();
    }

    for (auto ii = lo; ii < hi; ii++) {
      log_.progress(ii, lo, hi);
      auto const si = mapping_.sortedIndices[ii];
      auto const c = mapping_.cart[si];
      auto const nc = mapping_.noncart[si];
      auto const nck = noncart.chip(nc.spoke, 2).chip(nc.read, 1);
      kernel(mapping_.offset[si], pow(mapping_.sdc[si], DCexp_), k);
      stC.set(1, c.x - (InPlane / 2));
      stC.set(2, c.y - (InPlane / 2));
      if (safe_) {
        stC.set(3, c.z - (ThroughPlane / 2) - minZ[ti]);
        workspace[ti].slice(stC, szC) += nck.reshape(rshNC).broadcast(brdNC) *
                                         k.template cast<Cx>().reshape(rshC).broadcast(brdC);
      } else {
        stC.set(3, c.z - (ThroughPlane / 2));
        cart.slice(stC, szC) += nck.reshape(rshNC).broadcast(brdNC) *
                                k.template cast<Cx>().reshape(rshC).broadcast(brdC);
      }
    }
  };

  auto const &start = log_.now();
  cart.setZero();
  Threads::RangeFor(grid_task, mapping_.cart.size());
  if (safe_) {
    log_.info("Combining thread workspaces...");
    for (long ti = 0; ti < nThreads; ti++) {
      if (szZ[ti]) {
        cart.slice(
                Sz4{0, 0, 0, minZ[ti]},
                Sz4{cart.dimension(0), cart.dimension(1), cart.dimension(2), szZ[ti]})
            .device(dev) += workspace[ti];
      }
    }
  }
  log_.debug("Non-cart -> Cart: {}", log_.toNow(start));
}

template <int InPlane, int ThroughPlane>
void GridKB<InPlane, ThroughPlane>::A(Cx4 const &cart, Cx3 &noncart) const
{
  assert(noncart.dimension(0) == cart.dimension(0));
  assert(cart.dimension(1) == mapping_.cartDims[0]);
  assert(cart.dimension(2) == mapping_.cartDims[1]);
  assert(cart.dimension(3) == mapping_.cartDims[2]);

  long const nchan = cart.dimension(0);
  using FixZero = Eigen::type2index<0>;
  using FixOne = Eigen::type2index<1>;
  using FixIn = Eigen::type2index<InPlane>;
  using FixThrough = Eigen::type2index<ThroughPlane>;
  Eigen::IndexList<int, FixIn, FixIn, FixThrough> szC;
  szC.set(0, nchan);

  auto grid_task = [&](long const lo, long const hi) {
    Kernel k;
    Eigen::IndexList<FixZero, int, int, int> stC;
    for (auto ii = lo; ii < hi; ii++) {
      log_.progress(ii, lo, hi);
      auto const si = mapping_.sortedIndices[ii];
      auto const c = mapping_.cart[si];
      auto const nc = mapping_.noncart[si];
      kernel(mapping_.offset[si], 1.f, k);
      stC.set(1, c.x - (InPlane / 2));
      stC.set(2, c.y - (InPlane / 2));
      stC.set(3, c.z - (ThroughPlane / 2));
      noncart.chip(nc.spoke, 2).chip(nc.read, 1) = cart.slice(stC, szC).contract(
          k.template cast<Cx>(),
          Eigen::IndexPairList<
              Eigen::type2indexpair<1, 0>,
              Eigen::type2indexpair<2, 1>,
              Eigen::type2indexpair<3, 2>>());
    }
  };
  auto const &start = log_.now();
  noncart.setZero();
  Threads::RangeFor(grid_task, mapping_.cart.size());
  log_.debug("Cart -> Non-cart: {}", log_.toNow(start));
}

template struct GridKB<3, 3>;
template struct GridKB<3, 1>;