#include "../src/fft_plan.h"
#include "../src/log.h"
#include "../src/tensorOps.h"
#include <catch2/catch.hpp>
#include <fmt/format.h>

TEST_CASE("FFT", "[3D]")
{
  Log log;
  FFT::Start(log);

  auto sx = GENERATE(3, 5, 7, 32);
  auto sy = sx;
  SECTION("ThreeD")
  {
    auto sz = GENERATE(1, 2, 3, 16, 32);
    long const N = sx * sy * sz;
    Cx3 data(Sz3{sx, sy, sz});
    Cx3 ref(sx, sy, sz);
    FFT::ThreeD fft(data, log);

    ref.setConstant(1.f);
    data.setZero();
    data(sx / 2, sy / 2, sz / 2) = sqrt(N); // Parseval's theorem
    fft.forward(data);
    CHECK(Norm(data - ref) == Approx(0.f).margin(1.e-3f));
    fft.reverse(data);
    ref.setZero();
    ref(sx / 2, sy / 2, sz / 2) = sqrt(N);
    CHECK(Norm(data - ref) == Approx(0.f).margin(1.e-3f));
  }

  SECTION("Planned<5, 3>")
  {
    auto sz = GENERATE(1, 3, 7, 8, 16);
    long const N = sx * sy * sz;
    long const nc = 32;
    long const ne = 1;
    Cx5 data(nc, ne, sx, sy, sz);
    Cx5 ref(nc, ne, sx, sy, sz);
    FFT::Planned<5, 3> fft(data, log);

    ref.setConstant(1.f);
    data.setZero();
    data.chip(sz / 2, 4).chip(sy / 2, 3).chip(sx / 2, 2).setConstant(sqrt(N));
    fft.forward(data);
    CHECK(Norm(data - ref) == Approx(0.f).margin(1.e-6f * N * nc));
    fft.reverse(data);
    ref.setZero();
    ref.chip(sz / 2, 4).chip(sy / 2, 3).chip(sx / 2, 2).setConstant(sqrt(N));
    CHECK(Norm(data - ref) == Approx(0.f).margin(1.e-6f * N * nc));
  }

  FFT::End(log);
}
