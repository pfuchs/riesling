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
    float const retain,
    Log &log)
{
  log.info(FMT_STRING("ESPIRIT Calibration Size {} Kernel Size {}"), calSz, kSz);
  Cx4 lores = lo_gridder.newGrid();
  FFT3N fftGrid(lores, log);
  gridder.toCartesian(data, grid);
  Cx2 const kernels = ToKernels(grid, 7, 24, (2 * lo_gridder.info().read_gap + 1), log);

  auto const bdcsvd = kMat.transpose().bdcSvd(Eigen::ComputeThinV);
  long const nRetain = std::lrintf(retain * calTotal);
  Cx5 kRetain(nChan, kSz, kSz, kSz, nRetain);
  auto retainMap = CollapseToMatrix<Cx5, 4>(kRetain);
  log.info(FMT_STRING("Retaining {} singular vectors"), nRetain);
  retainMap = bdcsvd.matrixV().leftCols(nRetain);

  // As the Matlab reference version says, "rotate kernel to order by maximum variance"
  auto retainReshaped = retainMap.reshaped(nChan, kSz * kSz * kSz * nRetain);
  Eigen::MatrixXcf rotation = retainReshaped.transpose().bdcSvd(Eigen::ComputeFullV).matrixV();

  fmt::print(
      FMT_STRING("rR {} {} rotation {} {}\n"),
      retainReshaped.rows(),
      retainReshaped.cols(),
      rotation.rows(),
      rotation.cols());
  retainMap = (retainReshaped.transpose() * rotation)
                  .transpose()
                  .reshaped(nChan * kSz * kSz * kSz, nRetain);

  Cx4 tempKernel(nChan, calSz, calSz, calSz);
  Log nullLog;
  FFT3N fftKernel(tempKernel, nullLog);
  Cx5 imgKernels(nChan, calSz, calSz, calSz, nRetain);
  for (long ii = 0; ii < nRetain; ii++) {
    ZeroPad(kRetain.chip(ii, 4), tempKernel);
    fftKernel.reverse();
    imgKernels.chip(ii, 4) = tempKernel;
  }

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