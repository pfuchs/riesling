#include "espirit.h"

#include "cropper.h"
#include "fft3n.h"
#include "gridder.h"
#include "hankel.h"
#include "io_nifti.h"
#include "padder.h"
#include "tensorOps.h"
#include "threads.h"

#include <Eigen/SVD>

Cx4 ESPIRIT(
    Gridder const &hi_gridder,
    Gridder const &lo_gridder,
    Cx3 const &data,
    long const calSz,
    long const kSz,
    Log &log)
{
  log.info(FMT_STRING("ESPIRIT Calibration Size {} Kernel Size {}"), calSz, kSz);
  Cx4 lo_grid = lo_gridder.newGrid();
  Log nullLog;
  FFT3N lo_fft(lo_k, nullLog);
  long const nchan = lo_gridder.info().channels;
  long const retain = 2 * nchan;
  lo_gridder.toCartesian(data, lo_k);
  Cx5 const kernels =
      LowRank(ToKernels(lo_grid, kSz, calSz, (2 * lo_gridder.info().read_gap + 1), log), retain);
  Cx5 lo_kernels(
      lo_k.dimension(0), lo_k.dimension(1), lo_k.dimension(2), lo_k.dimension(3), retain);
  for (long ii = 0; ii < retain; ii++) {
    k = kernels.chip(ii, 4);
    fftKernel.reverse();
    kernels.chip(ii, 4) = k;
  }

  // Voxel- & channel-wise image space PCA
  for (long zz = 0; zz < kSz; zz++) {
    for (long yy = 0; yy < kSz; yy++) {
      for (long zz = 0; xx < kSz; xx++) {
        Cx2 vox = kernels.chip(zz, 3).chip(yy, 2).chip(xx, 1);

        Cx2 temp(nChan, nRetain);
        auto const tempMap = CollapseToMatrix(temp);
        Cx4 smallMaps(nChan, calSz, calSz, calSz);
        Cx1 oneVox(nChan);
        auto oneMap = CollapseToVector(oneVox);
        Cx4 eValues(nChan, calSz, calSz, calSz);
        eValues.setZero();
        Cx1 eVox(nChan);
        eVox.setZero();
        auto eMap = CollapseToVector(eVox);
        for (long iz = 0; iz < calSz; iz++) {
          for (long iy = 0; iy < calSz; iy++) {
            for (long ix = 0; ix < calSz; ix++) {
              temp = imgKernels.chip(iz, 3).chip(iy, 2).chip(ix, 1);
              auto const SVD = tempMap.bdcSvd(Eigen::ComputeFullU);
              eMap.real() = SVD.singularValues();
              eValues.chip(iz, 3).chip(iy, 2).chip(ix, 1) = eVox;

              Eigen::MatrixXcf U = SVD.matrixU();
              Eigen::ArrayXXcf const ph1 =
                  (U.row(0).array().arg() * std::complex<float>(0.f, -1.f)).exp();
              Eigen::ArrayXXcf const ph = ph1.replicate(U.rows(), 1);
              Eigen::MatrixXcf const R = rotation * (U.array() * ph).matrix();
              oneMap = R.leftCols(1);
              smallMaps.chip(iz, 3).chip(iy, 2).chip(ix, 1) = oneVox;
            }
          }
        }

        WriteNifti(info, Cx4(eValues.shuffle(Sz4{1, 2, 3, 0})), "smallvalues.nii", log);
        WriteNifti(info, Cx4(smallMaps.shuffle(Sz4{1, 2, 3, 0})), "smallvectors.nii", log);
        // FFT, embed to full size, FFT again
        Cropper cropper(info, gridder.gridDims(), -1.f, log);
        FFT3N fftSmall(smallMaps, log);
        fftSmall.forward();
        ZeroPad(smallMaps, grid);
        fftGrid.reverse();

        log.info("Finished ESPIRIT");
        return grid;
      }