#include "../../src/op/grid.h"
#include "../../src/sdc.h"
#include "../../src/tensorOps.h"
#include "../../src/traj_spirals.h"
#include "../../src/trajectory.h"

#include <catch2/catch.hpp>

TEST_CASE("ops-grid", "[ops]")
{
  Log log(Log::Level::Images);
  long const M = 16;
  Info const info{.type = Info::Type::ThreeD,
                  .channels = 1,
                  .matrix = Eigen::Array3l::Constant(M),
                  .read_points = M / 2,
                  .read_gap = 0,
                  .spokes_hi = long(M_PI * M * M),
                  .spokes_lo = 0,
                  .lo_scale = 1.f,
                  .volumes = 1,
                  .echoes = 1,
                  .tr = 1.f,
                  .voxel_size = Eigen::Array3f::Constant(1.f),
                  .origin = Eigen::Array3f::Constant(0.f),
                  .direction = Eigen::Matrix3f::Identity()};
  auto const points = ArchimedeanSpiral(info);
  Trajectory const traj(info, points, log);
  R2 const sdc = SDC::Pipe(traj, log);
  // With credit to PyLops
  SECTION("NN-Dot")
  {
    auto grid = make_grid(traj, 2.f, Kernels::NN, false, log);
    grid->setSDC(sdc);
    Cx3 y(info.channels, info.read_points, info.spokes_total()),
      v(info.channels, info.read_points, info.spokes_total());
    Cx4 x = grid->newMultichannel(1), u = grid->newMultichannel(1);

    v.setRandom();
    u.setRandom();

    grid->A(u, y);
    grid->Adj(v, x);

    auto const yy = Dot(y, v);
    auto const xx = Dot(u, x);
    log.image(R3(sdc.reshape(Sz3{1, sdc.dimension(0), sdc.dimension(1)})), "sdc.nii");
    log.image(y, "y.nii");
    log.image(v, "v.nii");
    log.image(x, "x.nii");
    log.image(u, "u.nii");
    CHECK(std::abs((yy - xx) / (yy + xx + 1.e-15f)) == Approx(0).margin(1.e-6));
    // "Dot squared"
    grid->A(x, y);
    log.image(y, "y2.nii");
    fmt::print(FMT_STRING("Norm(y) {} Norm(v) {} Norm(y - v) {}\n"), Norm(y), Norm(v), Norm(y - v));
    CHECK(Norm(y - v) == Approx(0).margin(1.e-6));
  }

  SECTION("KB3-Dot")
  {
    auto grid = make_grid(traj, 2.f, Kernels::KB3, false, log);
    grid->setSDC(sdc);
    Cx3 y(info.channels, info.read_points, info.spokes_total()),
      v(info.channels, info.read_points, info.spokes_total());
    Cx4 x = grid->newMultichannel(1), u = grid->newMultichannel(1);

    v.setRandom();
    u.setRandom();

    grid->A(u, y);
    grid->Adj(v, x);

    auto const yy = Dot(y, v);
    auto const xx = Dot(u, x);
    CHECK(std::abs((yy - xx) / (yy + xx + 1.e-15f)) == Approx(0).margin(1.e-6));
    // "Dot squared"
    grid->A(x, y);
    fmt::print(FMT_STRING("Norm(y) {} Norm(v) {} Norm(y - v) {}\n"), Norm(y), Norm(v), Norm(y - v));
    CHECK(Norm(y - v) == Approx(0).margin(1.e-6));
  }

  SECTION("KB5-Dot")
  {
    auto grid = make_grid(traj, 2.f, Kernels::KB5, false, log);
    grid->setSDC(sdc);
    Cx3 y(info.channels, info.read_points, info.spokes_total()),
      v(info.channels, info.read_points, info.spokes_total());
    Cx4 x = grid->newMultichannel(1), u = grid->newMultichannel(1);

    v.setRandom();
    u.setRandom();

    grid->A(u, y);
    grid->Adj(v, x);

    auto const yy = Dot(y, v);
    auto const xx = Dot(u, x);
    CHECK(std::abs((yy - xx) / (yy + xx + 1.e-15f)) == Approx(0).margin(1.e-6));
    // "Dot squared"
    grid->A(x, y);
    fmt::print(FMT_STRING("Norm(y) {} Norm(v) {} Norm(y - v) {}\n"), Norm(y), Norm(v), Norm(y - v));
    CHECK(Norm(y - v) == Approx(0).margin(1.e-6));
  }
}