#include "zinfandel.h"

#include "parse_args.h"
#include "tensorOps.h"
#include "threads.h"

#include <Eigen/SVD>
#include <complex>

// Helper Functions
Cx2 GrabSources(
    Cx3 const &ks,
    float const scale,
    std::vector<long> const &src_offsets,
    long const st_cal_read,
    long const n_cal_read,
    std::vector<long> const &spokes)
{
  assert(src_offsets.size());
  assert(
      (st_cal_read + n_cal_read + *std::max_element(src_offsets.begin(), src_offsets.end())) <
      ks.dimension(1));
  long const n_chan = ks.dimension(0);
  long const n_cal_spoke = spokes.size();
  long const n_src = src_offsets.size();
  Cx2 S(n_chan * n_src, n_cal_read * n_cal_spoke);
  S.setZero();
  auto const sc = S.slice(Sz2{0, 0}, Sz2{n_chan, 1}).constant(scale);
  for (long i_spoke = 0; i_spoke < n_cal_spoke; i_spoke++) {
    long const col_spoke = i_spoke * n_cal_read;
    long const ind_spoke = spokes[i_spoke];
    assert(ind_spoke < ks.dimension(2));
    for (long ir = 0; ir < n_cal_read; ir++) {
      long const col = col_spoke + ir;
      for (long is = 0; is < n_src; is++) {
        long const ind_read = st_cal_read + ir + src_offsets[is];
        long const row = is * n_chan;
        assert(ind_read >= 0);
        assert(ind_read < ks.dimension(1));
        S.slice(Sz2{row, col}, Sz2{n_chan, 1}) =
            ks.slice(Sz3{0, ind_read, ind_spoke}, Sz3{n_chan, 1, 1}).reshape(Sz2{n_chan, 1}) / sc;
      }
    }
  }
  return S;
}

Cx2 GrabTargets(
    Cx3 const &ks,
    float const scale,
    long const st_tgt_read,
    long const n_tgt_read,
    std::vector<long> const &spokes)
{
  long const n_chan = ks.dimension(0);
  long const n_spoke = spokes.size();
  Cx2 T(n_chan, n_tgt_read * n_spoke);
  T.setZero();
  for (long i_spoke = 0; i_spoke < n_spoke; i_spoke++) {
    long const col_spoke = i_spoke * n_tgt_read;
    long const ind_spoke = spokes[i_spoke];
    for (long i_read = 0; i_read < n_tgt_read; i_read++) {
      long const col = col_spoke + i_read;
      T.chip(col, 1) =
          ks.chip(ind_spoke, 2).chip(st_tgt_read + i_read, 1) / T.chip(col, 1).constant(scale);
    }
  }
  return T;
}

Eigen::MatrixXcf CalcWeights(Cx2 const &src, Cx2 const tgt, float const lambda)
{
  Eigen::Map<Eigen::MatrixXcf const> srcM(src.data(), src.dimension(0), src.dimension(1));
  Eigen::Map<Eigen::MatrixXcf const> tgtM(tgt.data(), tgt.dimension(0), tgt.dimension(1));

  if (lambda > 0.f) {
    auto const reg = lambda * srcM.norm() * Eigen::MatrixXcf::Identity(srcM.rows(), srcM.rows());
    auto const rhs = srcM * srcM.adjoint() + reg;
    Eigen::MatrixXcf pinv = rhs.completeOrthogonalDecomposition().pseudoInverse();
    return tgtM * srcM.adjoint() * pinv;
  } else {
    Eigen::MatrixXcf const pinv = srcM.completeOrthogonalDecomposition().pseudoInverse();
    return tgtM * pinv;
  }
}

std::vector<long>
FindClosest(R3 const &traj, long const &tgt, long const &n_spoke, std::vector<long> &all_spokes)
{
  std::vector<long> spokes(n_spoke);
  R1 const end_is = traj.chip(tgt, 2).chip(traj.dimension(1) - 1, 1);
  std::partial_sort(
      all_spokes.begin(),
      all_spokes.begin() + n_spoke,
      all_spokes.end(),
      [&traj, end_is](long const a, long const b) {
        auto const &end_a = traj.chip(a, 2).chip(traj.dimension(1) - 1, 1);
        auto const &end_b = traj.chip(b, 2).chip(traj.dimension(1) - 1, 1);
        return Norm(end_a - end_is) < Norm(end_b - end_is);
      });
  std::copy_n(all_spokes.begin(), n_spoke, spokes.begin());
  return spokes;
}

// Actual calculation
void zinfandel(
    long const gap_sz,
    long const n_src,
    long const n_cal_spoke,
    long const n_cal_read1,
    float const lambda,
    R3 const &traj,
    Cx3 &ks,
    Log &log)
{
  long const n_cal_read = n_cal_read1 < 1 ? ks.dimension(1) - (gap_sz + n_src) : n_cal_read1;

  log.info(
      FMT_STRING("ZINFANDEL Gap {} Sources {} Cal Spokes/Read {}/{} "),
      gap_sz,
      n_src,
      n_cal_spoke,
      n_cal_read);
  long const total_spokes = ks.dimension(2);
  float const scale = R0(ks.abs().maximum())();
  std::vector<long> srcs(n_src);
  std::iota(srcs.begin(), srcs.end(), 1);
  auto spoke_task = [&](long const spoke_lo, long const spoke_hi) {
    std::vector<long> spokes(n_cal_spoke);
    for (auto is = spoke_lo; is < spoke_hi; is++) {
      std::iota(
          spokes.begin(),
          spokes.end(),
          std::clamp(is - n_cal_spoke / 2, 0L, total_spokes - n_cal_spoke));
      auto const calS = GrabSources(ks, scale, srcs, gap_sz + 1, n_cal_read, spokes);
      auto const calT = GrabTargets(ks, scale, gap_sz, n_cal_read, spokes);
      auto const W = CalcWeights(calS, calT, lambda);

      for (long ig = gap_sz; ig > 0; ig--) {
        auto const S = GrabSources(ks, scale, srcs, ig, 1, {is});
        Eigen::Map<Eigen::MatrixXcf const> SM(S.data(), S.dimension(0), S.dimension(1));
        auto const T = W * SM;
        for (long icoil = 0; icoil < ks.dimension(0); icoil++) {
          ks(icoil, ig - 1, is) = T(icoil) * scale;
        }
      }
    }
  };
  Threads::RangeFor(spoke_task, 0, total_spokes);
}
