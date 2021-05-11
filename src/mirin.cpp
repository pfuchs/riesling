#include "mirin.h"

#include "cropper.h"
#include "fft3.h"
#include "fft3n.h"
#include "gridder.h"
#include "hankel.h"
#include "io_nifti.h"
#include "lowrank.h"
#include "padder.h"
#include "sdc.h"
#include "tensorOps.h"
#include "threads.h"

void MIRIN(
    Info const &in_info,
    R3 const &traj,
    float const os,
    Kernel *const kernel,
    std::string const &sdc,
    long const calSz,
    long const kSz,
    long const its,
    long const retain,
    Cx3 &in_data,
    Log &log)
{
  float const sense_res = 10.f;
  Gridder gridder(in_info, traj, os, kernel, log, sense_res, true);
  SDC::Load(sdc, in_info, traj, kernel, gridder, log);

  Cx4 x0 = gridder.newGrid();
  Cx4 x = gridder.newGrid();
  Cx4 y = gridder.newGrid();
  Cx3 transfer = gridder.newGrid1();
  FFT::Start(log);
  FFT3N fft4(x, log);
  FFT3 fft3(transfer, log);
  {
    Cx2 ones(in_info.read_points, in_info.spokes_total());
    ones.setConstant(1.f);
    gridder.toCartesian(ones, transfer);
  }
  log.image(transfer, "mirin-transfer.nii");
  long const nChan = x.dimension(0);
  Cx2 kernels(nChan * kSz * kSz * kSz, calSz * calSz * calSz);
  Eigen::Map<Eigen::MatrixXcf> kMat(kernels.data(), kernels.dimension(0), kernels.dimension(1));
  long const nRetain = retain * in_info.channels;
  if (nRetain > kMat.cols()) {
    log.fail(
        FMT_STRING(
            "Total cal size {} is smaller than number of vectors to retain {}, increase cal size"),
        kMat.cols(),
        nRetain);
  }
  log.info(
      FMT_STRING("SVD size {} {}, retaining {} singular vectors"),
      kMat.rows(),
      kMat.cols(),
      nRetain);

  fmt::print(
      FMT_STRING("kernels {} grid {}\n"),
      fmt::join(kernels.dimensions(), ","),
      fmt::join(x.dimensions(), ","));

  gridder.toCartesian(in_data, x0);
  log.image(x0, "mirin-x0.nii");
  fft4.reverse(x0);
  float const norm0 = Norm(x0);
  log.image(x0, "mirin-img0.nii");
  x.setZero();
  y.setZero();
  auto dev = Threads::GlobalDevice();
  for (long ii = 0; ii < its; ii++) {
    // Data consistency
    fft4.forward(y);
    y.device(dev) = y * Tile(transfer, in_info.channels);
    fft4.reverse(y);
    y.device(dev) = x - (y - x0);
    log.image(y, fmt::format(FMT_STRING("mirin-img-{:02d}.nii"), ii));
    fft4.forward(y);
    log.image(y, fmt::format(FMT_STRING("mirin-ks-{:02d}.nii"), ii));
    ToKernels(calSz, kSz, y, kernels, log);
    // log.image(
    //     Cx3(kernels.reshape(Sz3{kernels.dimension(0), kernels.dimension(1), 1})),
    //     fmt::format(FMT_STRING("mirin-kernels-pre{:02d}.nii"), ii));
    kMat = LowRank(kMat, nRetain);
    // log.image(
    //     Cx3(kernels.reshape(Sz3{kernels.dimension(0), kernels.dimension(1), 1})),
    //     fmt::format(FMT_STRING("mirin-kernels-post{:02d}.nii"), ii));
    FromKernels(calSz, kSz, kernels, y, log);
    log.image(y, fmt::format(FMT_STRING("mirin-kslrank-{:02d}.nii"), ii));
    fft4.reverse(y);
    log.image(y, fmt::format(FMT_STRING("mirin-imglrank-{:02d}.nii"), ii));
    float const delta = Norm(y - x) / norm0;
    log.info(FMT_STRING("MIRIN {} δ {}"), ii, delta);
    x.device(dev) = y;
    if (delta < 1.e-5) {
      break;
    }
  }
  log.image(x, "mirin-img-final.nii");
  fft4.forward(x);
  Cx3 out_data(in_data.dimensions());

  Info out_info = in_info;
  out_info.read_gap = 0;
  Gridder outGridder(out_info, traj, os, kernel, log, sense_res, true);
  outGridder.toNoncartesian(x, out_data);
  in_data.slice(Sz3{0, 0, 0}, Sz3{in_info.channels, in_info.read_gap, in_info.spokes_total()}) =
      out_data.slice(Sz3{0, 0, 0}, Sz3{in_info.channels, in_info.read_gap, in_info.spokes_total()});
  FFT::End(log);
}
