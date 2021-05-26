#include "espirit.h"

#include "cropper.h"
#include "decomp.h"
#include "fft3n.h"
#include "gridder.h"
#include "hankel.h"
#include "io_nifti.h"
#include "padder.h"
#include "tensorOps.h"
#include "threads.h"

Cx4 ESPIRIT(
    Gridder const &hi_gridder,
    Gridder const &lo_gridder,
    Cx3 const &data,
    long const calSz,
    long const kSz,
    Log &log)
{
  long const nchan = lo_gridder.info().channels;
  long const retain = 2 * nchan;
  log.info(FMT_STRING("ESPIRIT Calibration Size {} Kernel Size {}"), calSz, kSz);
  Cx4 lo_grid = lo_gridder.newGrid();
  FFT3N lo_fft(lo_grid, log);
  lo_grid.setZero();
  lo_gridder.toCartesian(data, lo_grid);
  log.image(lo_grid, "espirit-lo-grid.nii");
  log.info(FMT_STRING("Calculating k-space kernels"));
  Cx5 const mini_kernels = LowRank(ToKernels(lo_grid, kSz, calSz, 0, log), retain, log);
  log.image(Cx4(mini_kernels.chip(0, 4)), "espirit-mini-kernels-ks.nii");
  Cx5 lo_kernels(
      lo_grid.dimension(0),
      lo_grid.dimension(1),
      lo_grid.dimension(2),
      lo_grid.dimension(3),
      retain);
  lo_kernels.setZero();
  Cropper const lo_mini(
      Dims3{lo_grid.dimension(1), lo_grid.dimension(2), lo_grid.dimension(3)},
      Dims3{kSz, kSz, kSz},
      log);
  log.info(FMT_STRING("Transform to image kernels"));
  for (long kk = 0; kk < retain; kk++) {
    lo_mini.crop4(lo_grid) = mini_kernels.chip(kk, 4);
    lo_fft.reverse();
    lo_kernels.chip(kk, 4) = lo_grid;
    log.progress(kk, retain);
  }
  log.image(Cx4(lo_kernels.chip(0, 4)), "espirit-lo-kernels-img.nii");
  // Voxel- & channel-wise image space PCA, store matrix along leading dimension
  Cx4 cov(nchan * nchan, lo_grid.dimension(1), lo_grid.dimension(2), lo_grid.dimension(3));
  FFT3N cov_fft(cov, log);
  cov.setZero();
  log.info(FMT_STRING("Calculate image co-variance"));
  for (long zz = 0; zz < lo_kernels.dimension(3); zz++) {
    for (long yy = 0; yy < lo_kernels.dimension(2); yy++) {
      for (long xx = 0; xx < lo_kernels.dimension(1); xx++) {
        Cx2 const vox = lo_kernels.chip(zz, 3).chip(yy, 2).chip(xx, 1);
        Cx2 const gram =
            vox.conjugate().contract(vox, Eigen::IndexPairList<Eigen::type2indexpair<1, 1>>());
        cov.chip(zz, 3).chip(yy, 2).chip(xx, 1) = gram.reshape(Sz1{nchan * nchan});
      }
    }
    log.progress(zz, lo_kernels.dimension(3));
  }
  log.image(cov, "espirit-cov.nii");
  // Pad to full size and then do eigenmap calculation
  Cx4 hi = hi_gridder.newGrid();
  Cx4 hi_cov(nchan * nchan, hi.dimension(1), hi.dimension(2), hi.dimension(3));
  FFT3N hi_fft(hi_cov, log);
  hi_cov.setZero();
  Cropper const cov_cropper(
      Dims3{hi.dimension(1), hi.dimension(2), hi.dimension(3)},
      Dims3{cov.dimension(1), cov.dimension(2), cov.dimension(3)},
      log);
  log.info(FMT_STRING("Upsample co-variance"));
  cov_fft.forward(cov);
  cov_cropper.crop4(hi_cov) = cov;
  hi_fft.reverse(hi_cov);
  log.image(hi_cov, "espirit-hi-cov.nii");
  log.info(FMT_STRING("Extract eigenvectors"));
  for (long zz = 0; zz < hi_cov.dimension(3); zz++) {
    for (long yy = 0; yy < hi_cov.dimension(2); yy++) {
      for (long xx = 0; xx < hi_cov.dimension(1); xx++) {
        Cx2 const gram = hi_cov.chip(zz, 3).chip(yy, 2).chip(xx, 1).reshape(Sz2{nchan, nchan});
        Cx2 vecs(gram.dimensions());
        Cx1 vals(gram.dimension(0));
        PCA(gram, vecs, vals);
        hi.chip(zz, 3).chip(yy, 2).chip(xx, 1) = vecs.chip(nchan - 1, 1);
      }
    }
    log.progress(zz, hi_cov.dimension(3));
  }

  log.info("Finished ESPIRIT");
  return hi;
}