#include "pinot.h"

#include "cropper.h"
#include "fft3.h"
#include "fft3n.h"
#include "gridder.h"
#include "nlcg.hpp"
#include "tensorOps.h"
#include "threads.h"

void PINOT(
    Info const &in_info,
    R3 const &traj,
    float const os,
    Kernel *const kernel,
    Cx3 &in_data,
    Log &log)
{
  auto dev = Threads::GlobalDevice();
  float const sense_res = 12.f;
  log.info("Using PINOT-GRIS");

  long gap = in_info.read_gap;
  Info info = in_info;
  info.read_gap = 0;

  Cx3 data = in_data;
  Gridder gridder(info, traj, os, kernel, log, sense_res, true);
  gridder.setSDCExponent(0.8f);
  Cx4 grid = gridder.newGrid();
  FFT3N fft(grid, log);
  grid.setZero();

  Sz3 const gridDims = gridder.gridDims();
  Cx3 img(gridDims);
  FFT3 fftImg(img, log);
  Sz4 const senseDims{info.channels, gridDims[0], gridDims[1], gridDims[2]};
  Sz4 const combiDims{info.channels + 1, senseDims[1], senseDims[2], senseDims[3]};
  Cx4 sense(senseDims);
  sense.setZero();
  Cx4 combined(combiDims);
  combined.setZero();

  Cx3 W(gridDims);
  for (long zz = 0; zz < gridDims[0]; zz++) {
    float const z = float(zz) / gridDims[0] - 0.5f;
    for (long yy = 0; yy < gridDims[1]; yy++) {
      float const y = float(yy) / gridDims[1] - 0.5f;
      for (long xx = 0; xx < gridDims[2]; xx++) {
        float const x = float(xx) / gridDims[2] - 0.5f;
        float kr = (x * x + y * y + z * z);
        W(xx, yy, zz) = pow(1.f + kr * kr, -32.f); // Uecker et al gives l=16
      }
    }
  }
  log.image(W, "pinot-W.nii");

  log.info("Resid between {}-{}", info.read_gap, gridder.maxRead());
  auto reg = [&](Cx4 &x, bool const adjoint) {
    static Cx3 temp(data.dimensions());
    fft.forward(x);
    if (adjoint) {
      x.device(dev) = x * Tile(W, info.channels).conjugate();
    } else {
      x.device(dev) = x * Tile(W, info.channels);
    }
    fft.reverse(x);
  };

  // Fixed image, variable sensitivities
  NLCG<4, 3>::Residual res = [&](Cx4 const &x, Cx3 &y) {
    sense.device(dev) = x.slice(Sz4{1, 0, 0, 0}, senseDims);
    // reg(sense, false);
    grid.device(dev) = sense * TileToMatch(x.chip(0, 0), senseDims);
    fft.forward(grid);
    gridder.toNoncartesian(grid, y);
    y.slice(Sz3{0, gap, 0}, Sz3{info.channels, gridder.maxRead() - gap, info.spokes_total()}) =
        y.slice(Sz3{0, gap, 0}, Sz3{info.channels, gridder.maxRead() - gap, info.spokes_total()}) -
        data.slice(
            Sz3{0, gap, 0}, Sz3{info.channels, gridder.maxRead() - gap, info.spokes_total()});
    return Norm2(y.slice(
               Sz3{0, gap, 0}, Sz3{info.channels, gridder.maxRead() - gap, info.spokes_total()})) /
           2.f;
  };

  long it = 0;
  NLCG<4, 3>::JacAdjoint jac = [&](Cx4 const &x, Cx3 const &y, Cx4 &j) {
    gridder.toCartesian(y, grid);
    fft.reverse(grid);
    sense.device(dev) = x.slice(Sz4{1, 0, 0, 0}, senseDims);
    // reg(sense, false);
    j.chip(0, 0).device(dev) = (grid * sense.conjugate()).sum(Sz1{0});

    sense.device(dev) = grid * TileToMatch(x.chip(0, 0).conjugate(), senseDims);
    // reg(sense, true);
    j.slice(Sz4{1, 0, 0, 0}, senseDims) = sense;
    fft.forward(sense);
    img = j.chip(0, 0);
    fftImg.forward(img);
    it++;
  };

  combined.chip(0, 0).setConstant(1.f);
  combined.slice(Sz4{1, 0, 0, 0}, senseDims).setZero();

  float const dnorm = Norm2(data);
  float const cnorm = Norm2(combined.chip(0, 0));
  log.info(FMT_STRING("dnorm {} cnorm {}"), dnorm, cnorm);

  float const scale = sqrt(Norm2(data) / Norm2(combined.chip(0, 0)));
  data.device(dev) = data / data.constant(scale);
  log.info(FMT_STRING("nnorm {}"), Norm2(data));

  for (gap = in_info.read_gap; gap > 0; gap--) {
    data.slice(
        Sz3{0, in_info.read_gap, 0},
        Sz3{info.channels, in_info.read_points - in_info.read_gap, info.spokes_total()}) =
        in_data.slice(
            Sz3{0, in_info.read_gap, 0},
            Sz3{info.channels, in_info.read_points - in_info.read_gap, info.spokes_total()}) /
        in_data
            .slice(
                Sz3{0, in_info.read_gap, 0},
                Sz3{info.channels, in_info.read_points - in_info.read_gap, info.spokes_total()})
            .constant(scale);
    // data.slice(Sz3{0, 0, 0}, Sz3{info.channels, gap, 0}).setZero();
    log.image(
        Cx3(data.slice(Sz3{0, 0, 0}, Sz3{info.channels, 32, 32})),
        fmt::format("pinot-data-{}.nii", gap));
    NLCG<4, 3>::Run(res, jac, 8, 1.e-5f, data.dimensions(), combined, log);
    grid.setZero();
    grid.device(dev) =
        combined.slice(Sz4{1, 0, 0, 0}, senseDims) * TileToMatch(combined.chip(0, 0), senseDims);
    gridder.toNoncartesian(grid, data);
  }

  in_data.slice(Sz3{0, 0, 0}, Sz3{info.channels, in_info.read_gap, info.spokes_total()}) =
      data.slice(Sz3{0, 0, 0}, Sz3{info.channels, in_info.read_gap, info.spokes_total()});

  data.device(dev) = data * data.constant(scale);
  log.info("Finished PINOT");
}
