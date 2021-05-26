#include "../src/hankel.h"
#include <catch2/catch.hpp>
#include <fmt/ostream.h>

TEST_CASE("Hankel", "[Hankel]")
{
  Log log;

  long const nchan = 2;
  long const gridSz = 8;
  Cx4 grid(nchan, gridSz, gridSz, gridSz);
  grid.chip(0, 0).setConstant(1.f);
  grid.chip(1, 0).setConstant(2.f);
  long const kSz = 3;
  long const calSz = 5;

  SECTION("No Gap")
  {
    Cx5 k = ToKernels(grid, kSz, calSz, 0, log);

    CHECK(k.dimension(0) == nchan);
    CHECK(k.dimension(1) == kSz);
    CHECK(k.dimension(2) == kSz);
    CHECK(k.dimension(3) == kSz);
    CHECK(k.dimension(4) == (calSz * calSz * calSz));
    CHECK(k(0, 0, 0, 0, 0) == 1.f);
    CHECK(k(1, 0, 0, 0, 0) == 2.f);
  }

  SECTION("Gap")
  {
    long const gap = 1;
    grid.chip(4, 3).chip(4, 2).chip(4, 1).setZero();
    Cx5 k = ToKernels(grid, kSz, calSz, gap, log);
    CHECK(k.dimension(0) == nchan);
    CHECK(k.dimension(1) == kSz);
    CHECK(k.dimension(2) == kSz);
    CHECK(k.dimension(3) == kSz);
    long const gSz = gap + 2 * (kSz / 2);
    CHECK(k.dimension(4) == (calSz * calSz * calSz) - (gSz * gSz * gSz));
    CHECK(k(0, 0, 0, 0, 0) == 1.f);
    CHECK(k(1, 0, 0, 0, 0) == 2.f);
    CHECK(B0((k.real() > k.real().constant(0.f)).all())());
  }
}