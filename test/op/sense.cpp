#include "../../src/op/sense.hpp"
#include "../../src/tensorOps.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

using namespace rl;
using namespace Catch;

TEST_CASE("ops-sense")
{
  Index const channels = 2, mapSz = 4;
  // With credit to PyLops
  SECTION("Dot Test")
  {
    Cx4 x(1, mapSz, mapSz, mapSz), u(1, mapSz, mapSz, mapSz);
    Cx5 maps(channels, 1, mapSz, mapSz, mapSz);
    Cx5 y(channels, 1, mapSz, mapSz, mapSz), v(channels, 1, mapSz, mapSz, mapSz);

    v.setRandom();
    u.setRandom();
    // The maps need to be normalized for the Dot test
    maps.setRandom();
    Cx4 const rss = ConjugateSum(maps, maps).sqrt();
    maps = maps / Tile(rss, channels);

    SenseOp sense(maps, 1);
    y = sense.forward(u);
    x = sense.adjoint(v);

    auto const yy = Dot(y, v);
    auto const xx = Dot(u, x);
    CHECK(std::abs((yy - xx) / (yy + xx + 1.e-15f)) == Approx(0).margin(1.e-6));
  }
}