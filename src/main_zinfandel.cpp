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
      parser, "KERNEL SIZE", "Mirin Kernel size (default 7)", {"kernel"}, 7);
  args::ValueFlag<long> calSz(
      parser, "CAL SIZE", "MIRIN Calibration region size (default 31)", {"cal"}, 25);
  args::ValueFlag<long> its(parser, "ITERATIONS", "Maximum iterations", {"its"});
  args::ValueFlag<float> retain(
      parser,
      "RETAIN",
      "MIRIN Fraction of singular vectors to retain (default 0.25)",
      {"retain"},
      0.25);

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
    rad_ks.slice(Sz3{0, 0, 0}, Sz3{info.channels, gap_sz, info.spokes_total()})
        .setZero(); // Ensure no rubbish
    if (twostep) {
      zinfandel2(gap_sz, src.Get(), read.Get(), l1.Get(), rad_ks, log);
    } else if (mirin) {
      Kernel *kernel =
          kb ? (Kernel *)new KaiserBessel(3, osamp.Get(), (info.type == Info::Type::ThreeD))
             : (Kernel *)new NearestNeighbour();
      MIRIN(
          info,
          trajectory,
          osamp.Get(),
          kernel,
          calSz.Get(),
          kernelSz.Get(),
          its.Get(),
          retain.Get(),
          rad_ks,
          log);
    } else if (pinot) {
      Kernel *kernel =
          kb ? (Kernel *)new KaiserBessel(3, osamp.Get(), (info.type == Info::Type::ThreeD))
             : (Kernel *)new NearestNeighbour();
      PINOT(info, trajectory, osamp.Get(), kernel, rad_ks, log);
    } else {
      zinfandel(gap_sz, src.Get(), spokes.Get(), read.Get(), l1.Get(), rad_ks, log);
    }
    if (pw && rbw) {
      slab_correct(out_info, pw.Get(), rbw.Get(), vol, log);
    }
    rad_ks.chip(iv, 3);
  }
  writer.writeNoncartesian(rad_ks);
  log.info("Finished");
  return EXIT_SUCCESS;
}
