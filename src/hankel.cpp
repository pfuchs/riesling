#include "hankel.h"

#include "tensorOps.h"

void ToKernels(long const blkSz, long const kSz, Cx4 const &grid, Cx2 &kernels, Log &log)
{
  long const nchan = grid.dimension(0);
  long const gridHalf = grid.dimension(1) / 2;
  long const blkHalf = blkSz / 2;
  long const kHalf = kSz / 2;
  if (grid.dimension(1) < (blkSz + kSz - 1)) {
    log.fail(
        FMT_STRING("Grid size {} not large enough for block size {} + kernel size {}"),
        grid.dimension(1),
        blkSz,
        kSz);
  }
  long const rows = nchan * kSz * kSz * kSz;
  long const cols = blkSz * blkSz * blkSz;
  assert(kernels.dimension(0) == rows);
  assert(kernels.dimension(1) == cols);

  long col = 0;
  long const st = gridHalf - blkHalf - kHalf;
  for (long iz = 0; iz < blkSz; iz++) {
    long const st_z = st + iz;
    for (long iy = 0; iy < blkSz; iy++) {
      long const st_y = st + iy;
      for (long ix = 0; ix < blkSz; ix++) {
        long const st_x = st + ix;
        kernels.chip(col, 1) =
            grid.slice(Sz4{0, st_x, st_y, st_z}, Sz4{nchan, kSz, kSz, kSz}).reshape(Sz1{rows});
        col++;
      }
    }
  }
  assert(col == cols);
}

void FromKernels(long const blkSz, long const kSz, Cx2 const &kernels, Cx4 &grid, Log &log)
{
  long const nchan = grid.dimension(0);
  long const gridHalf = grid.dimension(1) / 2;
  long const blkHalf = blkSz / 2;
  long const kHalf = kSz / 2;
  if (grid.dimension(1) < (blkSz + kSz - 1)) {
    log.fail(
        FMT_STRING("Grid size {} not large enough for block size {} + kernel size {}"),
        grid.dimension(1),
        blkSz,
        kSz);
  }
  long const rows = nchan * kSz * kSz * kSz;
  long const cols = blkSz * blkSz * blkSz;
  assert(kernels.dimension(0) == rows);
  assert(kernels.dimension(1) == cols);

  long const st = gridHalf - blkHalf - kHalf;
  long const sz = blkSz + kSz - 1;
  R3 count(sz, sz, sz);
  count.setZero();
  Cx4 data(nchan, sz, sz, sz);
  data.setZero();
  long col = 0;
  for (long iz = 0; iz < blkSz; iz++) {
    for (long iy = 0; iy < blkSz; iy++) {
      for (long ix = 0; ix < blkSz; ix++) {
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