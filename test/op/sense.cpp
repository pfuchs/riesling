#include "../../src/op/sense.h"
#include "../../src/tensorOps.h"
#include <catch2/catch.hpp>
#include <fmt/ostream.h>

TEST_CASE("ops-sense", "[ops]")
{
  long const channels = 2, mapSz = 4, gridSz = 6;

  SECTION("Dot Test")
  {
    Cx3 img(mapSz, mapSz, mapSz), img2(mapSz, mapSz, mapSz);
    Cx4 maps(channels, mapSz, mapSz, mapSz), grid(channels, gridSz, gridSz, gridSz),
        grid2(channels, gridSz, gridSz, gridSz);

    img.setRandom();
    grid.setRandom();
    // The maps need to be normalized for the Dot test
    maps.setRandom();
    Cx3 rss = (maps * maps.conjugate()).sum(Sz1{0}).sqrt();
    maps = maps / Tile(rss, channels);

    SenseOp sense(maps, grid.dimensions());
    sense.A(img, grid2);
    sense.Adj(grid, img2);

    fmt::print("img\n{}\nimg2\n{}\ngrid\n{}\ngrid2\n{}\n", img, img2, grid, grid2);
    CHECK(Dot(img, img2) == Dot(grid2, grid));
  }
}