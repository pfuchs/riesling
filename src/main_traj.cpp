#include "types.h"

#include "fft3.h"
#include "gridder.h"
#include "io_hd5.h"
#include "io_nifti.h"
#include "log.h"
#include "parse_args.h"
#include "threads.h"
#include <complex>

int main_traj(args::Subparser &parser)
{
  args::Positional<std::string> fname(parser, "FILE", "HD5 file to calculate trajectory from");
  args::ValueFlag<std::string> oname(parser, "OUTPUT", "Override output name", {"out", 'o'});
  args::ValueFlag<float> osamp(
      parser, "GRID OVERSAMPLE", "Oversampling factor for gridding, default 2", {'g', "grid"}, 2.f);
  args::Flag est_dc(parser, "ESTIMATE DC", "Estimate DC weights instead of analytic", {"est_dc"});
  args::Flag stack(parser, "STACK", "Trajectory is stack-of-stars or similar", {"stack"});
  args::Flag kb(parser, "KB", "Use Kaiser-Bessel interpolation", {"kb"});
  Log log = ParseCommand(parser, fname);
  FFT::Start(log);
  HD5Reader reader(fname.Get(), log);
  auto const &info = reader.info();
  Kernel *kernel =
      kb ? (Kernel *)new KaiserBessel(3, osamp.Get(), !stack) : (Kernel *)new NearestNeighbour();
  Gridder gridder(info, reader.readTrajectory(), osamp.Get(), est_dc, kernel, stack, log);
  Cx3 grid = gridder.newGrid1();
  FFT3 fft(grid, log);

  grid.setZero();
  Cx2 rad_ks(info.read_points, info.spokes_total());
  rad_ks.setConstant(1.0f);
  gridder.toCartesian(rad_ks, grid);
  WriteNifti(info, R3(grid.abs()), OutName(fname, oname, "traj"), log);
  fft.reverse();
  WriteNifti(info, grid, OutName(fname, oname, "psf"), log);
  FFT::End(log);
  return EXIT_SUCCESS;
}
