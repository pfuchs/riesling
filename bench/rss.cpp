#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include "../src/threads.h"
#include "../src/types.h"
#include <catch2/catch.hpp>
#include <fmt/ostream.h>

TEST_CASE("RSS")
{
  Cx4 grid(16, 128, 128, 128);
  Cx3 image(128, 128, 128);

  image.device(Threads::GlobalDevice()) =
      (cropper.crop4(grid) * cropper.crop4(grid).conjugate()).sum(Sz1{0}).sqrt();
}