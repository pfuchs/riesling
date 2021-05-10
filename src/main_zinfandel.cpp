#include "io_hd5.h"
#include "io_nifti.h"
#include "kernels.h"
#include "log.h"
#include "mirin.h"
#include "parse_args.h"
#include "pinot.h"
#include "slab_correct.h"
#include "threads.h"
#include "types.h"
#include "zinfandel.h"
#include <filesystem>

int main_zinfandel(args::Subparser &parser)
{
  COMMON_RECON_ARGS;

  args::ValueFlag<long> gap(
      parser, "DEAD-TIME GAP", "Set gap value (default use header value)", {'g', "gap"}, -1);
  args::ValueFlag<long> src(
      parser, "SOURCES", "Number of ZINFANDEL sources (default 4)", {"src"}, 4);
  args::ValueFlag<long> spokes(
      parser, "CAL SPOKES", "Number of spokes to use for calibration (default 5)", {"spokes"}, 5);
  args::ValueFlag<long> read(
      parser, "CAL READ", "Read calibration size (default all)", {"read"}, 0);
  args::ValueFlag<float> l1(
      parser, "LAMBDA", "Tikhonov regularization (default 0)", {"lambda"}, 0.f);
  args::ValueFlag<float> pw(
      parser, "PULSE WIDTH", "Pulse-width for slab profile correction", {"pw"}, 0.f);
  args::ValueFlag<float> rbw(
      parser, "BANDWIDTH", "Read-out bandwidth for slab profile correction (kHz)", {"rbw"}, 0.f);
  args::Flag twostep(parser, "TWOSTEP", "Use two step method", {"two", '2'});
  args::Flag pinot(parser, "PINOT", "Use PINOT", {"pinot"});
  args::Flag mirin(parser, "MIRIN", "Use MIRIN", {"mirin"});
  args::ValueFlag<long> kernelSz(
      parser, "KERNEL SIZE", "Mirin Kernel size (default 5)", {"kernel"}, 5);
  args::ValueFlag<long> calSz(
      parser, "CAL SIZE", "MIRIN Calibration region radius (default 16)", {"cal"}, 16);
  args::ValueFlag<long> its(parser, "ITERATIONS", "Maximum iterations", {"its"}, 4);
  args::ValueFlag<long> retain(
      parser,
      "RETAIN",
      "MIRIN Singular vectors per channel to retain (default 16)",
      {"retain"},
      16);

  Log log = ParseCommand(parser, fname);

  HD5::Reader reader(fname.Get(), log);
  auto info = reader.info();
  auto const trajectory = reader.readTrajectory();
  long const gap_sz = gap ? gap.Get() : info.read_gap;
  R3 traj = reader.readTrajectory();

  auto out_info = info;
  out_info.read_gap = 0;
  if (volume) {
    out_info.volumes = 1;
  }

  HD5::Writer writer(OutName(fname, oname, "zinfandel", "h5"), log);
  writer.writeInfo(out_info);
  writer.writeTrajectory(traj);
  writer.writeMeta(reader.readMeta());

  Cx4 rad_ks = info.noncartesianSeries();
  reader.readNoncartesian(rad_ks);
  for (auto const &iv : WhichVolumes(volume.Get(), info.volumes)) {
    Cx3 vol = rad_ks.chip(iv, 3);
    vol.slice(Sz3{0, 0, 0}, Sz3{info.channels, gap_sz, info.spokes_total()})
        .setZero(); // Ensure no rubbish
    if (twostep) {
      zinfandel2(gap_sz, src.Get(), read.Get(), l1.Get(), vol, log);
    } else if (mirin) {
      Kernel *kernel =
          kb ? (Kernel *)new KaiserBessel(3, osamp.Get(), (info.type == Info::Type::ThreeD))
             : (Kernel *)new NearestNeighbour();
      MIRIN(
          info,
          trajectory,
          osamp.Get(),
          kernel,
          sdc.Get(),
          calSz.Get(),
          kernelSz.Get(),
          its.Get(),
          retain.Get(),
          vol,
          log);
    } else if (pinot) {
      Kernel *kernel =
          kb ? (Kernel *)new KaiserBessel(3, osamp.Get(), (info.type == Info::Type::ThreeD))
             : (Kernel *)new NearestNeighbour();
      PINOT(info, trajectory, osamp.Get(), kernel, vol, log);
    } else {
      zinfandel(gap_sz, src.Get(), spokes.Get(), read.Get(), l1.Get(), vol, log);
    }
    if (pw && rbw) {
      slab_correct(out_info, pw.Get(), rbw.Get(), vol, log);
    }
    rad_ks.chip(iv, 3) = vol;
  }
  writer.writeNoncartesian(rad_ks);
  log.info("Finished");
  return EXIT_SUCCESS;
}
