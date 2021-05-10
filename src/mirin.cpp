#include "mirin.h"

#include "cropper.h"
#include "fft3.h"
#include "fft3n.h"
#include "gridder.h"
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
    long const calRad,
    long const kSz,
    long const its,
    long const retain,
    Cx3 &in_data,
    Log &log)
{
  Info info = in_info;
  info.read_gap = 0;
  float const sense_res = 8.f;
  Gridder gridder(info, traj, 1, kernel, log, sense_res, true);
  SDC::Load(sdc, info, traj, kernel, gridder, log);
  long const gap = in_info.read_gap;

  Cx4 grid = gridder.newGrid();
  Cx4 grid0 = gridder.newGrid();
  Cx4 temp = gridder.newGrid();
  Cx3 transfer = gridder.newGrid1();
  FFT::Start(log);
  FFT3N fft4(grid, log);
  FFT3 fft3(transfer, log);
  Cx2 ones(info.read_points, info.spokes_total());
  gridder.toCartesian(ones, transfer);

  long const gridHalf = grid.dimension(1) / 2;
  long const kernelHalf = kSz / 2;
  long const calHalf = calRad - kernelHalf;
  long const calSz = 2 * calHalf + 1;
  long const nChan = grid.dimension(0);
  long const calTotal = calSz * calSz * calSz;
  long const startPoint = gridHalf - calHalf - kernelHalf;
  log.info(FMT_STRING("MIRIN Kernel Size {} Cal Rad {} Cal Sz {}"), kSz, calRad, calSz);
  if (startPoint < 0) {
    log.fail("MIRIN Calibration + Kernel Size exceeds grid size");
  }

  Cx5 kernels(nChan, kSz, kSz, kSz, calTotal);
  auto kMat = CollapseToMatrix<Cx5, 4>(kernels);
  long const nRetain = retain * info.channels;
  log.info(
      FMT_STRING("SVD size {} {}, retaining {} out of {} singular vectors"),
      kMat.rows(),
      kMat.cols(),
      nRetain,
      calTotal);

  auto toKernels = [&](Cx4 const &grid, Cx5 &kernels) -> long {
    long col = 0;
    for (long iz = -calHalf; iz <= calHalf; iz++) {
      for (long iy = -calHalf; iy <= calHalf; iy++) {
        for (long ix = -calHalf; ix <= calHalf; ix++) {
          long const st_z = gridHalf + iz;
          long const st_y = gridHalf + iy;
          long const st_x = gridHalf + ix;
          kernels.chip(col, 4) = grid.slice(Sz4{0, st_x, st_y, st_z}, Sz4{nChan, kSz, kSz, kSz});
          col++;
        }
      }
    }
    return col;
  };

  auto fromKernels = [&](Cx5 const &kernels, Cx4 &grid) -> long {
    static R3 count(gridder.gridDims());
    count.setZero();
    grid.setZero();
    long col = 0;
    for (long iz = -calHalf; iz <= calHalf; iz++) {
      for (long iy = -calHalf; iy <= calHalf; iy++) {
        for (long ix = -calHalf; ix <= calHalf; ix++) {
          long const st_z = startPoint + iz;
          long const st_y = startPoint + iy;
          long const st_x = startPoint + ix;
          grid.slice(Sz4{0, st_x, st_y, st_z}, Sz4{nChan, kSz, kSz, kSz}) += kernels.chip(col, 4);
          count.slice(Sz3{st_x, st_y, st_z}, Sz3{kSz, kSz, kSz}) +=
              count.slice(Sz3{st_x, st_y, st_z}, Sz3{kSz, kSz, kSz}).constant(1.f);
          col++;
        }
      }
    }
    grid = grid.abs().select(grid / Tile(count, info.channels).cast<Cx>(), grid);
    return col;
  };

  fmt::print(
      FMT_STRING("kernels {} grid {}\n"),
      fmt::join(kernels.dimensions(), ","),
      fmt::join(grid.dimensions(), ","));

  gridder.toCartesian(in_data, grid0);
  fft4.reverse(grid0);
  float const norm0 = Norm(grid0);
  log.image(grid0, "mirin-img0.nii");
  grid.setZero();
  auto dev = Threads::GlobalDevice();
  for (long ii = 0; ii < its; ii++) {
    // Data consistency
    temp.device(dev) = grid;
    fft4.forward(temp);
    temp.device(dev) = temp * Tile(transfer, info.channels);
    fft4.reverse(temp);
    temp.device(dev) = grid - (temp - grid0);
    log.image(temp, fmt::format(FMT_STRING("mirin-img-{:02d}.nii"), ii));
    fft4.forward(temp);
    log.image(temp, fmt::format(FMT_STRING("mirin-grid-{:02d}.nii"), ii));
    toKernels(temp, kernels);
    kMat = LowRank(kMat, nRetain);
    fromKernels(kernels, temp);
    log.image(temp, fmt::format(FMT_STRING("mirin-lrank-{:02d}.nii"), ii));
    fft4.reverse(temp);
    log.image(temp, fmt::format(FMT_STRING("mirin-lrank-img-{:02d}.nii"), ii));
    float const delta = Norm(temp - grid) / norm0;
    log.info(FMT_STRING("MIRIN {} δ {}"), ii, delta);
    grid.device(dev) = temp;
    if (delta < 1.e-5) {
      break;
    }
  }
  log.image(grid, "mirin-img-final.nii");
  fft4.forward(grid);
  Cx3 out_data(in_data.dimensions());
  gridder.toNoncartesian(grid, out_data);
  in_data.slice(Sz3{0, 0, 0}, Sz3{info.channels, gap, info.spokes_total()}) =
      out_data.slice(Sz3{0, 0, 0}, Sz3{info.channels, gap, info.spokes_total()});
  FFT::End(log);
}
