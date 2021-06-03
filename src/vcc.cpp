#include "vcc.h"

#include "tensorOps.h"

void VCC(Cx4 &data, Log &log)
{
  long const nc = data.dimension(0);
  long const nx = data.dimension(1);
  long const ny = data.dimension(2);
  long const nz = data.dimension(3);
  Cx3 phase(nx, ny, nz);

  for (long iz = 1; iz < nz; iz++) {
    for (long iy = 1; iy < ny; iy++) {
      for (long ix = 1; ix < nx; ix++) {
        Cx1 const vals = data.chip(iz, 3).chip(iy, 2).chip(ix, 1);
        Cx1 const cvals = data.chip(nz - iz, 3).chip(ny - iy, 2).chip(nx - ix, 1);
        float const p = std::log(Dot(cvals, vals)).imag() / 2.f;
        phase(ix, iy, iz) = std::polar(1.f, p);
      }
    }
  }
  log.image(phase, "vcc-correction.nii");
  log.info("Applying Virtual Conjugate Coil phase correction");
  data = data * Tile(phase, nc);
}
