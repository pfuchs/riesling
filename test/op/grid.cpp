#include "../../src/op/gridBase.hpp"
#include "../../src/sdc.h"
#include "../../src/tensorOps.h"
#include "../../src/traj_spirals.h"
#include "../../src/trajectory.h"

#include <catch2/catch.hpp>

using namespace rl;

TEST_CASE("Grid Basic", "[grid]")
{
  // Log::SetLevel(Log::Level::Debug);
  SECTION("Grid Size") {
    Index const M = GENERATE(7,15,31);
    Info const info{
      .type = Info::Type::ThreeD, .matrix = Eigen::Array3l(M,M,M), .channels = 1, .read_points = 3, .spokes = 1};
    Re3 points(3, 3, 1);
    points.setZero();
    points(0, 0, 0) = -0.4f;
    points(0, 2, 0) = 0.4f;
    Trajectory const traj(info, points);

    float const osamp = GENERATE(2.f,2.7f,3.f);
    auto const kernel = rl::make_kernel("FI5", info.type, osamp);
    Mapping const mapping(traj, kernel.get(), osamp);
    auto grid = make_grid<Cx>(kernel.get(), mapping, info.channels);
    Cx3 ks(grid->outputDimensions());
    Cx5 img(grid->inputDimensions());
    ks.setConstant(1.f);
    img = grid->Adj(ks);
    CHECK(Norm(img) == Approx(std::sqrt(ks.size())).margin(5.e-2f));
    ks = grid->A(img);
    CHECK(Norm(img) == Approx(std::sqrt(ks.size())).margin(5.e-2f));
  }
}

TEST_CASE("ops-grid")
{
  Index const M = 16;
  float const os = 2.f;
  Info const info{
    .type = Info::Type::ThreeD,
    .matrix = Eigen::Array3l::Constant(M),
    .channels = 1,
    .read_points = Index(os * M / 2),
    .spokes = Index(M * M),
    .volumes = 1,
    .frames = 1,
    .tr = 1.f,
    .voxel_size = Eigen::Array3f::Constant(1.f),
    .origin = Eigen::Array3f::Constant(0.f),
    .direction = Eigen::Matrix3f::Identity()};
  auto const points = ArchimedeanSpiral(info.read_points, info.spokes);
  Trajectory const traj(info, points);
  auto const nn = rl::make_kernel("NN", info.type, os);
  Mapping const mapping(traj, nn.get(), os);
  auto grid = make_grid<Cx>(nn.get(), mapping, 1);

  SDCOp sdc(SDC::Pipe(traj, true, os), info.channels);
  auto const dims = grid->inputDimensions();
  Cx5 x(dims), y(dims);

  /* I don't think the classic Dot test from PyLops is applicable to gridding,
   * because it would not be correct to have a random radial k-space. The k0
   * samples should all be the same, not random. Hence instead I calculate
   * y = Adj*A*x for NN and then check if Dot(x,y) is the same as Dot(y,y).
   * Can't check for y = x because samples not on the radial spokes will be
   * missing, and checking for KB kernel is not valid because the kernel blurs
   * the grid.
   */
  SECTION("SDC-Full")
  {
    x.setRandom();
    y = grid->Adj(sdc.Adj(grid->A(x)));
    auto const xy = Dot(x, y);
    auto const yy = Dot(y, y);
    CHECK(std::abs((yy - xy) / (yy + xy + 1.e-15f)) == Approx(0).margin(1.e-6));
  }
}