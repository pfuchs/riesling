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

Cx4 ESPIRIT(Gridder const &gridder, Cx3 const &data, long const calSz, long const kSz, Log &log)
{
  long const nchan = gridder.info().channels;
  long const retain = 16 * nchan;
  log.info(FMT_STRING("ESPIRIT Calibration Size {} Kernel Size {}"), calSz, kSz);
  Cx4 lo_grid = gridder.newGrid();
  FFT3N lo_fft(lo_grid, log);
  lo_grid.setZero();
  gridder.toCartesian(data, lo_grid);
  log.image(lo_grid, "espirit-lo-grid.nii");
  log.info(FMT_STRING("Calculating k-space kernels"));
  long const gap = gridder.info().read_gap ? gridder.info().read_gap * 2 + 1 : 0;
  Cx5 const mini_kernels = LowRank(ToKernels(lo_grid, kSz, calSz, gap, log), retain, log);
  log.image(Cx4(mini_kernels.chip(0, 4)), "espirit-mini-kernel0-ks.nii");
  log.image(Cx4(mini_kernels.chip(retain / 2, 4)), "espirit-mini-kernel1-ks.nii");
  log.image(Cx4(mini_kernels.chip(retain - 1, 4)), "espirit-mini-kernel2-ks.nii");
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
    lo_grid.setZero();
    lo_mini.crop4(lo_grid) = mini_kernels.chip(kk, 4);
    lo_fft.reverse(lo_grid);
    lo_kernels.chip(kk, 4) = lo_grid;
    log.progress(kk, retain);
  }
  log.image(Cx4(lo_kernels.chip(0, 4)), "espirit-lo-kernel0-img.nii");
  log.image(Cx4(lo_kernels.chip(retain / 2, 4)), "espirit-lo-kernel1-img.nii");
  log.image(Cx4(lo_kernels.chip(retain - 1, 4)), "espirit-lo-kernel2-img.nii");
  // Voxel- & channel-wise image space PCA, store matrix along leading dimension
  Cx4 cov(nchan * nchan, lo_grid.dimension(1), lo_grid.dimension(2), lo_grid.dimension(3));
  FFT3N cov_fft(cov, log);
  cov.setZero();
  log.info(FMT_STRING("Calculate image co-variance"));
  Cx const scale(kSz * kSz * kSz);
  for (long zz = 0; zz < lo_kernels.dimension(3); zz++) {
    for (long yy = 0; yy < lo_kernels.dimension(2); yy++) {
      for (long xx = 0; xx < lo_kernels.dimension(1); xx++) {
        Cx2 const vox = lo_kernels.chip(zz, 3).chip(yy, 2).chip(xx, 1) / scale;
        cov.chip(zz, 3).chip(yy, 2).chip(xx, 1) = Covariance(vox).reshape(Sz1{nchan * nchan});
      }
    }
    log.progress(zz, lo_kernels.dimension(3));
  }
  log.image(cov, "espirit-cov.nii");
  log.info(FMT_STRING("Extract eigenvectors"));
  Cx4 vec = gridder.newGrid();
  Cx4 val = gridder.newGrid();
  for (long zz = 0; zz < cov.dimension(3); zz++) {
    for (long yy = 0; yy < cov.dimension(2); yy++) {
      for (long xx = 0; xx < cov.dimension(1); xx++) {
        Cx2 const vox_cov = cov.chip(zz, 3).chip(yy, 2).chip(xx, 1).reshape(Sz2{nchan, nchan});
        Cx2 vecs(vox_cov.dimensions());
        R1 vals(vox_cov.dimension(0));
        PCA(vox_cov, vecs, vals);
        vec.chip(zz, 3).chip(yy, 2).chip(xx, 1) = vecs.chip(nchan - 1, 1).conjugate();
        val.chip(zz, 3).chip(yy, 2).chip(xx, 1) = vals.cast<Cx>();
      }
    }
    log.progress(zz, cov.dimension(3));
  }
  log.image(val, "espirit-val.nii");
  log.image(vec, "espirit-vec.nii");
  log.info("Finished ESPIRIT");
  return vec;
}