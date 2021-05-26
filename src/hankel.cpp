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
  long const gapK = gapSz ? gapSz + 2 * kHalf : 0;
  long const nk = calSz * calSz * calSz - (gapK * gapK * gapK);
  Cx5 kernels(nchan, kSz, kSz, kSz, nk);

  long k = 0;
  long nskip = 0;
  long const gapSt = calHalf - gapHalf - kHalf;
  long const gapEnd = calHalf + gapHalf + kHalf;

  long const st = gridHalf - calHalf - kHalf;
  log.info(FMT_STRING("Hankelfying {} kernels"), nk);
  for (long iz = 0; iz < calSz; iz++) {
    long const st_z = st + iz;
    for (long iy = 0; iy < calSz; iy++) {
      long const st_y = st + iy;
      for (long ix = 0; ix < calSz; ix++) {
        if (gapSz && (ix >= gapSt && ix <= gapEnd) && (iy >= gapSt && iy <= gapEnd) &&
            (iz >= gapSt && iz <= gapEnd)) {
          nskip++;
          continue;
        }
        long const st_x = st + ix;
        Sz4 sst{0, st_x, st_y, st_z};
        Sz4 ssz{nchan, kSz, kSz, kSz};
        kernels.chip(k, 4) = grid.slice(sst, ssz);
        k++;
      }
    }
  }
  assert(k == nk);
  log.image(Cx4(kernels.chip(0, 4)), "hankel-kernel0.nii");
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
  assert(kernels.dimension(0) == nchan * kSz * kSz * kSz);
  assert(kernels.dimension(1) == calSz * calSz * calSz);

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
  assert(col == calSz * calSz * calSz);
  grid.slice(Sz4{0, st, st, st}, Sz4{nchan, sz, sz, sz}) =
      data.abs().select(data / Tile(count, nchan).cast<Cx>(), data);
}