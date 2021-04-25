#include "mirin.h"

#include "cropper.h"
#include "fft3n.h"
#include "gridder.h"
#include "io_nifti.h"
#include "padder.h"
#include "tensorOps.h"
#include "threads.h"

#include <Eigen/SVD>

void MIRIN(
    Info const &in_info,
    R3 const &traj,
    float const os,
    Kernel *const kernel,
    long const calSz,
    long const kSz,
    long const its,
    float const retain,
    Cx3 &in_data,
    Log &log)
{
  float const sense_res = 12.f;
  log.info(FMT_STRING("MIRIN Calibration Size {} Kernel Size {}"), calSz, kSz);
  Info info = in_info;
  info.read_gap = 0;
  Gridder gridder(info, traj, os, SDC::Analytic, kernel, log, sense_res, false);
  long const gap = in_info.read_gap;
  long const read_sz = gridder.maxRead() - gap;
  Cx4 grid = gridder.newGrid();
  FFT3N fft(grid, log);
  grid.setZero();

  long const gridHalf = grid.dimension(1) / 2;
  long const calHalf = calSz / 2;
  long const kernelHalf = kSz / 2;
  long const nChan = grid.dimension(0);
  long const calTotal = calSz * calSz * calSz;
  long const startPoint = gridHalf - calHalf - kernelHalf;
  if (startPoint < 0) {
    log.fail("MIRIN Calibration + Kernel Size exceeds grid size");
  }
  auto toKernels = [&](long const rad, Cx3 const &data, Cx5 &kernels) -> long {
    gridder.toCartesian(data, grid);
    long col = 0;
    for (long iz = -calHalf; iz <= calHalf; iz++) {
      for (long iy = -calHalf; iy <= calHalf; iy++) {
        for (long ix = -calHalf; ix <= calHalf; ix++) {
          if (sqrt(iz * iz + iy * iy + ix * ix) >= rad) {
            long const st_z = gridHalf + iz;
            long const st_y = gridHalf + iy;
            long const st_x = gridHalf + ix;
            kernels.chip(col, 4) = grid.slice(Sz4{0, st_x, st_y, st_z}, Sz4{nChan, kSz, kSz, kSz});
          }
          col++;
        }
      }
    }
    return col;
  };

  auto fromKernels = [&](long const rad, Cx5 const &kernels, Cx3 &data) -> long {
    static R3 count(gridder.gridDims());
    count.setZero();
    grid.setZero();
    long col = 0;
    for (long iz = -calHalf; iz <= calHalf; iz++) {
      for (long iy = -calHalf; iy <= calHalf; iy++) {
        for (long ix = -calHalf; ix <= calHalf; ix++) {
          if (sqrt(iz * iz + iy * iy + ix * ix) >= rad) {
            long const st_z = startPoint + iz;
            long const st_y = startPoint + iy;
            long const st_x = startPoint + ix;
            grid.slice(Sz4{0, st_x, st_y, st_z}, Sz4{nChan, kSz, kSz, kSz}) += kernels.chip(col, 4);
            count.slice(Sz3{st_x, st_y, st_z}, Sz3{kSz, kSz, kSz}) +=
                count.slice(Sz3{st_x, st_y, st_z}, Sz3{kSz, kSz, kSz}).constant(1.f);
          }
          col++;
        }
      }
    }
    grid = grid.abs().select(grid / Tile(count, info.channels).cast<Cx>(), grid);
    gridder.toNoncartesian(grid, data);
    return col;
  };

  Cx3 data = in_data;
  data.slice(Sz3{0, 0, 0}, Sz3{info.channels, gap, info.spokes_total()}).setZero();
  Cx3 data1 = data;
  float const norm0 =
      Norm(data.slice(Sz3{0, 0, 0}, Sz3{info.channels, gap + read_sz, info.spokes_total()}));
  long maxCols = calSz * calSz * calSz;
  Cx5 kernels(nChan, kSz, kSz, kSz, maxCols);
  auto kMat = CollapseToMatrix<Cx5, 4>(kernels);
  long const nRetain = std::lrintf(retain * calTotal);
  long const n_zeros = calTotal - nRetain;
  log.info(
      FMT_STRING("SVD size {} {}, retaining {} singular vectors"),
      kMat.rows(),
      kMat.cols(),
      nRetain);

  fmt::print(
      FMT_STRING("kernels {} grid {}\n"),
      fmt::join(kernels.dimensions(), ","),
      fmt::join(grid.dimensions(), ","));

  long rad = gap - 1;
  for (long ii = 0; ii < its; ii++) {
    long const toCols = toKernels(rad, data, kernels);
    assert(toCols <= maxCols);
    log.image(
        Cx3(data.slice(Sz3{0, 0, 0}, Sz3{info.channels, gridder.maxRead(), 64})),
        fmt::format(FMT_STRING("mirin-data-pre-{:02d}.nii"), ii));
    log.image(grid, fmt::format(FMT_STRING("mirin-grid-pre-{:02d}.nii"), ii));
    auto const svd = kMat.leftCols(toCols).bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::VectorXf vals = svd.singularValues();
    fmt::print("vals {}\n", vals.transpose());
    vals.tail(n_zeros).setZero();
    kMat.leftCols(toCols) = svd.matrixU() * vals.asDiagonal() * svd.matrixV().adjoint();
    long const fromCols = fromKernels(rad, kernels, data);
    assert(toCols == fromCols);
    float const delta =
        Norm((data - data1).slice(Sz3{0, 0, 0}, Sz3{info.channels, gap, info.spokes_total()})) /
        norm0;
    log.info(
        FMT_STRING("MIRIN {} δ {} norm0 {} num {}"),
        ii,
        delta,
        norm0,
        Norm((data - data1)
                 .slice(Sz3{0, 0, 0}, Sz3{info.channels, gap + read_sz, info.spokes_total()})));
    data1 = data;
    data.slice(Sz3{0, gap, 0}, Sz3{info.channels, read_sz, info.spokes_total()}) =
        in_data.slice(Sz3{0, gap, 0}, Sz3{info.channels, read_sz, info.spokes_total()});
    if (delta < 1.e-5) {
      break;
    }
    rad = std::max(0L, rad - 1);
  }
  in_data.slice(Sz3{0, 0, 0}, Sz3{info.channels, gap, info.spokes_total()}) =
      data.slice(Sz3{0, 0, 0}, Sz3{info.channels, gap, info.spokes_total()});
}
