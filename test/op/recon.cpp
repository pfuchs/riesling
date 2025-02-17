#include "../src/op/recon.hpp"
#include "log.hpp"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

using namespace rl;
using namespace Catch;

TEST_CASE("Recon", "[recon]")
{
  Log::SetLevel(Log::Level::Testing);
  Index const M = GENERATE(7, 15, 16);
  Index const nC = 4;
  Index const nF = 1;
  Info const info{.matrix = Sz3{M, M, M}};
  Re3 points(3, 3, 1);
  points.setZero();
  points(0, 0, 0) = -0.4f;
  points(1, 0, 0) = -0.4f;
  points(0, 2, 0) = 0.4f;
  points(1, 2, 0) = 0.4f;
  Trajectory const traj(info, points);

  float const osamp = GENERATE(2.f, 2.7f, 3.f);
  std::string const ktype = GENERATE("ES7");
  Re2 basis(nF, nF);
  basis.setConstant(1.f);
  auto nufft = make_nufft(traj, ktype, osamp, nC, traj.matrix(), basis);

  Cx5 senseMaps(AddFront(traj.matrix(), nC, nF));
  senseMaps.setConstant(std::sqrt(1./nC));
  auto sense = std::make_shared<SenseOp>(senseMaps, nF);
  Compose<SenseOp, TensorOperator<Cx, 5, 4>> recon(sense, nufft);

  Cx4 ks(recon.oshape);
  Cx4 img(recon.ishape);
  ks.setConstant(1.f);
  img = recon.adjoint(ks);
  // Super loose tolerance
  CHECK(Norm(img) == Approx(Norm(ks)).margin(2.e-1f));
  ks = recon.forward(img);
  CHECK(Norm(ks) == Approx(Norm(img)).margin(2.e-1f));
}
