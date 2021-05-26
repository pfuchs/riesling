#include "hankel.h"

#include "tensorOps.h"

Cx5 ToKernels(Cx4 const &grid, long const kSz, long const calSz, long const gapSz, Log &log)
{
  long const nchan = grid.dimension(0);
  long const gridHalf = grid.dimension(1) / 2;
  long const calHalf = calSz / 2;
  long const kHalf = kSz / 2;
  if (grid.dimension(1) < (calSz + kSz - 1)) {
    log.fail(
        FMT_STRING("Grid size {} not large enough for block size {} + kernel size {}"),
        grid.dimension(1),
        calSz,
        kSz);
  }
  long const gapHalf = gapSz / 2;
  long const gapK = gapSz + 2 * kHalf;
  long const rows = nchan * (kSz * kSz * kSz - gapK * gapK * gapK);
  long const nk = calSz * calSz * calSz;
  Cx5 kernels(nchan, kSz, kSz, kSz, nk);

  long l = 0;
  long const st = gridHalf - calHalf - kHalf;
  long const gapSt = calHalf - gapHalf - kHalf;
  long const gapEnd = calHalf + gapHalf + kHalf;
  for (long iz = 0; iz < calSz; iz++) {
    if (gapSz && ((iz < gapSt) || (iz > gapEnd))) {
      long const st_z = st + iz;
      for (long iy = 0; iy < calSz; iy++) {
        if (gapSz && ((iy < gapSt) || (iy > gapEnd))) {
          long const st_y = st + iy;
          for (long ix = 0; ix < calSz; ix++) {
            if (gapSz && ((ix < gapSt) || (ix > gapEnd))) {
              long const st_x = st + ix;
              kernels.chip(k, 4) = grid.slice(Sz4{0, st_x, st_y, st_z}, Sz4{nchan, kSz, kSz, kSz});
              k++;
            }
          }
        }
      }
    }
  }
  assert(k == nk);
  return kernels;
}

void FromKernels(long const calSz, long const kSz, Cx2 const &kernels, Cx4 &grid, Log &log)
{
  long const nchan = grid.dimension(0);
  long const gridHalf = grid.dimension(1) / 2;
  long const calHalf = calSz / 2;
  long const kHalf = kSz / 2;
  if (grid.dimension(1) < (calSz + kSz - 1)) {
    log.fail(
        FMT_STRING("Grid size {} not large enough for block size {} + kernel size {}"),
        grid.dimension(1),
        calSz,
        kSz);
  }
  long const rows = nchan * kSz * kSz * kSz;
  long const cols = calSz * calSz * calSz;
  assert(kernels.dimension(0) == rows);
  assert(kernels.dimension(1) == cols);

  long const st = gridHalf - calHalf - kHalf;
  long const sz = calSz + kSz - 1;
  R3 count(sz, sz, sz);
  count.setZero();
  Cx4 data(nchan, sz, sz, sz);
  data.setZero();
  long col = 0;
  for (long iz = 0; iz < calSz; iz++) {
    for (long iy = 0; iy < calSz; iy++) {
      for (long ix = 0; ix < calSz; ix++) {
        data.slice(Sz4{0, ix, iy, iz}, Sz4{nchan, kSz, kSz, kSz}) +=
            kernels.chip(col, 1).reshape(Sz4{nchan, kSz, kSz, kSz});
        count.slice(Sz3{ix, iy, iz}, Sz3{kSz, kSz, kSz}) +=
            count.slice(Sz3{ix, iy, iz}, Sz3{kSz, kSz, kSz}).constant(1.f);
        col++;
      }
    }
  }
  assert(col == cols);
  grid.slice(Sz4{0, st, st, st}, Sz4{nchan, sz, sz, sz}) =
      data.abs().select(data / Tile(count, nchan).cast<Cx>(), data);
}