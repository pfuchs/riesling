#include "mirin.h"

#include "tensorOps.h"
#include "threads.h"

#include <Eigen/SVD>
#include <complex>

void mirin(
    long const gap_sz,
    long const n_src_read,
    long const n_read1,
    long const n_src_spokes,
    long const its,
    float const frac,
    Cx3 &ks,
    Log &log)
{
  long const n_chan = ks.dimension(0);
  long const n_read = n_read1 < 1 ? ks.dimension(1) - (gap_sz + n_src_read) : n_read1;
  long const total_spokes = ks.dimension(2);

  long const n_rows = n_chan * n_src_read * n_src_spokes;
  long const n_cols = (gap_sz + n_read) * total_spokes;
  long const n_keep = std::max<long>(frac * std::min(n_rows, n_cols), 1L);
  long const n_zeros = std::min(n_rows, n_cols) - n_keep;
  log.info(
      FMT_STRING("ZINFANDEL Gap {} Sources {} Cal Read {} A Size {}x{} Keep {}"),
      gap_sz,
      n_src_read,
      n_read,
      n_rows,
      n_cols,
      n_keep);

  Cx2 At(n_rows, n_cols);
  At.setZero();
  Eigen::Map<Eigen::MatrixXcf> A(At.data(), n_rows, n_cols);

  // for (auto ii = gap_sz; ii > 0; ii--) {
  //   ks.chip(ii - 1, 1) = ks.chip(ii, 1);
  // }

  float const scale = R0(ks.abs().maximum())();
  auto toA = [&]() {
    long icol = 0;
    for (long is = 0; is < total_spokes; is++) {
      long const lo_spoke = std::clamp(is, 0L, total_spokes - n_src_spokes);
      for (long ii = 0; ii < (gap_sz + n_read); ii++) {
        Cx1 const data = ks.slice(Sz3{0, ii, lo_spoke}, Sz3{n_chan, n_src_read, n_src_spokes})
                             .reshape(Sz1{n_rows});
        At.chip(icol, 1) = (data.abs() > 0.f).select(data / data.constant(scale), At.chip(icol, 1));
        icol++;
      }
    }
    assert(icol == n_cols);
  };

  auto fromA = [&]() {
    ks.slice(Sz3{0, 0, 0}, Sz3{n_chan, gap_sz, total_spokes}).setZero(); // Ensure no rubbish
    long icol = 0;
    for (long is = 0; is < total_spokes; is++) {
      long const lo_spoke = std::clamp(is, 0L, total_spokes - n_src_spokes);
      for (long ii = 0; ii < gap_sz; ii++) {
        auto const blk = At.chip(icol, 1).reshape(Sz3{n_chan, n_src_read, n_src_spokes});
        ks.slice(Sz3{0, ii, lo_spoke}, Sz3{n_chan, gap_sz - ii, n_src_spokes}) +=
            blk.slice(Sz3{0, 0, 0}, Sz3{n_chan, gap_sz - ii, n_src_spokes}) *
            blk.slice(Sz3{0, 0, 0}, Sz3{n_chan, gap_sz - ii, n_src_spokes}).constant(scale);
        icol++;
      }
      icol += n_read;
    }
    assert(icol == n_cols);
    for (auto ii = 0; ii < gap_sz; ii++) {
      ks.chip(ii, 1) /= ks.chip(ii, 1).constant((ii + 1) * n_src_spokes);
    }

    log.info("Averaging k0");
    auto const k0cs = ks.chip(0, 1).cumsum(1);
    long const win = 16;
    ks.chip(0, 1).slice(Sz2{0, 0}, Sz2{n_chan, total_spokes - win}) =
        (k0cs.slice(Sz2{0, win}, Sz2{n_chan, total_spokes - win}) -
         k0cs.slice(Sz2{0, 0}, Sz2{n_chan, total_spokes - win})) /
        k0cs.slice(Sz2{0, 0}, Sz2{n_chan, total_spokes - win}).constant(win);
  };

  Cx3 oldGap = ks.slice(Sz3{0, 0, 0}, Sz3{n_chan, gap_sz, total_spokes});
  Cx4 history(n_chan, 32, 32, its);
  for (long ii = 0; ii < its; ii++) {
    toA();
    Cx3 reshape = At.slice(Sz2{0, 0}, Sz2{n_rows, 128}).reshape(Sz3{n_rows, 128, 1});
    log.image(reshape, fmt::format(FMT_STRING("mirin-{:02d}-A-pre.nii"), ii));
    auto const svd = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::VectorXf vals = svd.singularValues();
    long const n_zero = vals.size() - vals.array().log10().isFinite().count();
    float const biggest = vals(0);
    float const smallest = vals.tail(1)(0);
    vals.tail(n_zeros).setZero();
    A = svd.matrixU() * vals.asDiagonal() * svd.matrixV().adjoint();
    fromA();
    reshape = At.slice(Sz2{0, 0}, Sz2{n_rows, 128}).reshape(Sz3{n_rows, 128, 1});
    log.image(reshape, fmt::format(FMT_STRING("mirin-{:02d}-A-post.nii"), ii));
    auto const delta = Norm(oldGap - ks.slice(Sz3{0, 0, 0}, Sz3{n_chan, gap_sz, total_spokes}));
    log.info(
        FMT_STRING("MIRIN {} δ {} SVs {} High {} Low {} Ratio {} N=0 {}"),
        ii,
        delta,
        vals.size(),
        biggest,
        smallest,
        (biggest / smallest),
        n_zero);
    oldGap = ks.slice(Sz3{0, 0, 0}, Sz3{n_chan, gap_sz, total_spokes});
    history.chip(ii, 3) = ks.slice(Sz3{0, 0, 0}, Sz3{n_chan, 32, 32});
  }
  log.image(history, fmt::format(FMT_STRING("mirin-history.nii")));
}
