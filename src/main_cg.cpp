#include "types.h"

#include "cg.hpp"
#include "filter.h"
#include "log.h"
#include "op/recon.h"
#include "parse_args.h"
#include "sense.h"
#include "tensorOps.h"
#include "threads.h"

int main_cg(args::Subparser &parser)
{
  COMMON_RECON_ARGS;
  COMMON_SENSE_ARGS;

  args::ValueFlag<float> thr(
      parser, "TRESHOLD", "Threshold for termination (1e-10)", {"thresh"}, 1.e-10);
  args::ValueFlag<long> its(
      parser, "MAX ITS", "Maximum number of iterations (8)", {'i', "max_its"}, 8);
  args::ValueFlag<float> iter_fov(
      parser, "ITER FOV", "Iterations FoV in mm (default 256 mm)", {"iter_fov"}, 256);

  Log log = ParseCommand(parser, iname);
  FFT::Start(log);

  HD5::Reader reader(iname.Get(), log);
  Trajectory const traj = reader.readTrajectory();
  Info const &info = traj.info();
  Cx3 rad_ks = info.noncartesianVolume();

  long currentVolume = -1;
  Cx4 senseMaps;
  if (senseFile) {
    senseMaps = LoadSENSE(senseFile.Get(), log);
  } else {
    currentVolume = LastOrVal(senseVolume, info.volumes);
    reader.readNoncartesian(currentVolume, rad_ks);
    senseMaps = DirectSENSE(traj, osamp.Get(), kb, iter_fov.Get(), rad_ks, senseLambda.Get(), log);
  }

  ReconOp recon(traj, osamp.Get(), kb, fastgrid, sdc.Get(), senseMaps, log);
  recon.setPreconditioning(sdc_exp.Get());
  Cx3 vol(recon.dimensions());
  Cropper out_cropper(info, vol.dimensions(), out_fov.Get(), log);
  Cx3 cropped = out_cropper.newImage();
  Cx4 out = out_cropper.newSeries(info.volumes);
  auto const &all_start = log.now();
  for (long iv = 0; iv < info.volumes; iv++) {
    auto const &vol_start = log.now();
    if (iv != currentVolume) { // For single volume images, we already read it for senseMaps
      reader.readNoncartesian(iv, rad_ks);
      currentVolume = iv;
    }
    recon.Adj(rad_ks, vol); // Initialize
    cg(its.Get(), thr.Get(), recon, vol, log);
    cropped = out_cropper.crop3(vol);
    if (tukey_s || tukey_e || tukey_h) {
      ImageTukey(tukey_s.Get(), tukey_e.Get(), tukey_h.Get(), cropped, log);
    }
    out.chip(iv, 3) = cropped;
    log.info("Volume {}: {}", iv, log.toNow(vol_start));
  }
  log.info("All Volumes: {}", log.toNow(all_start));
  WriteOutput(out, mag, false, info, iname.Get(), oname.Get(), "cg", oftype.Get(), log);
  FFT::End(log);
  return EXIT_SUCCESS;
}
