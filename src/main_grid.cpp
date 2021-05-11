#include "types.h"

#include "gridder.h"
#include "io_hd5.h"
#include "io_nifti.h"
#include "log.h"
#include "parse_args.h"
#include "threads.h"
#include <filesystem>

int main_grid(args::Subparser &parser)
{
  COMMON_RECON_ARGS;
  args::ValueFlag<long> gap(parser, "GAP", "Set dead-time gap to value", {'g', "gap"});
  args::ValueFlag<float> res(parser, "RES", "Effective resolution to grid to", {'r', "res"}, -1.f);
  args::Flag shrink(parser, "SHRINK", "Shrink the grid to match the resolution", {'s', "shrink"});
  args::Flag forward(parser, "REV", "Apply forward gridding (to non-cartesian)", {'f', "fwd"});
  args::Flag nii(parser, "NII", "Output nifti", {'n', "nii"});
  Log log = ParseCommand(parser, fname);
  HD5::Reader reader(fname.Get(), log);
  auto info = reader.info();
  if (gap) {
    info.read_gap = gap.Get();
  }
  auto const trajectory = reader.readTrajectory();
  Kernel *kernel =
      kb ? (Kernel *)new KaiserBessel(kw.Get(), osamp.Get(), (info.type == Info::Type::ThreeD), log)
         : (Kernel *)new NearestNeighbour(kw ? kw.Get() : 1, log);
  Gridder gridder(info, trajectory, osamp.Get(), kernel, log, res.Get(), shrink);
  SDC::Load(sdc.Get(), info, trajectory, kernel, gridder, log);
  gridder.setSDCExponent(sdc_exp.Get());
  Cx3 rad_ks = info.noncartesianVolume();
  Cx4 grid = gridder.newGrid();

  long const vol = volume ? volume.Get() : 0;
  auto const &vol_start = log.now();

  if (forward) {
    reader.readCartesian(grid);
    gridder.toNoncartesian(grid, rad_ks);
  } else {
    reader.readNoncartesian(vol, rad_ks);
    gridder.toCartesian(rad_ks, grid);
  }

  if (nii) {
    WriteNifti(Info(), SwapToChannelLast(grid), OutName(fname, oname, "grid", "nii"), log);
  } else {
    HD5::Writer writer(OutName(fname, oname, "grid", "h5"), log);
    writer.writeInfo(info);
    writer.writeTrajectory(trajectory);
    if (forward) {
      writer.writeNoncartesian(
          rad_ks.reshape(Sz4{rad_ks.dimension(0), rad_ks.dimension(1), rad_ks.dimension(2), 1}));
      log.info("Wrote non-cartesian k-space. Took {}", log.toNow(vol_start));
    } else {
      writer.writeCartesian(grid);
      log.info("Wrote cartesian k-space. Took {}", log.toNow(vol_start));
    }
  }

  return EXIT_SUCCESS;
}
