#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include "../src/op/grid.h"
#include "../src/info.h"
#include "../src/kernel_nn.h"
#include "../src/op/grid_nn.h"
#include "../src/traj_archimedean.h"

#include <catch2/catch.hpp>

TEST_CASE("Grid")
{
  Log log;
  long const M = 64;
  long const C = 8;
  Info const info{
      .type = Info::Type::ThreeD,
      .channels = C,
      .matrix = Eigen::Array3l::Constant(M),
      .read_points = M / 2,
      .read_gap = 0,
      .spokes_hi = M * M,
      .spokes_lo = 0,
      .lo_scale = 1.f,
      .volumes = 1,
      .echoes = 1,
      .tr = 1.f,
      .voxel_size = Eigen::Array3f::Constant(1.f),
      .origin = Eigen::Array3f::Constant(0.f),
      .direction = Eigen::Matrix3f::Identity()};
  auto const points = ArchimedeanSpiral(info);
  Trajectory traj(info, points, log);

  float const os = 2.f;
  NearestNeighbour kernel(1);

  BENCHMARK("Mapping")
  {
    traj.mapping(os, kernel.radius());
  };

  GridOp gridder(traj.mapping(os, kernel.radius()), &kernel, false, log);
  GridNN gridnn(traj.mapping(os, kernel.radius()), false, log);
  auto nc = info.noncartesianVolume();
  auto c = gridder.newMultichannel(C);

  BENCHMARK("Noncartesian->Cartesian")
  {
    gridder.Adj(nc, c);
  };
  BENCHMARK("Cartesian->Noncartesian")
  {
    gridder.A(c, nc);
  };
  BENCHMARK("NN Noncartesian->Cartesian")
  {
    gridnn.Adj(nc, c);
  };
  BENCHMARK("NN Cartesian->Noncartesian")
  {
    gridnn.A(c, nc);
  };
}