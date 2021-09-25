#include "types.h"

#include "io_hd5.h"
#include "kernels.h"
#include "log.h"
#include "op/grid.h"
#include "parse_args.h"

int main_sdc(args::Subparser &parser)
{
  CORE_RECON_ARGS;

  Log log = ParseCommand(parser, iname);
  HD5::Reader reader(iname.Get(), log);
  auto const traj = reader.readTrajectory();
  auto const &info = traj.info();

  R2 dc;
  if (sdc.Get() == "pipe") {
    Kernel *kernel =
        kb ? (Kernel *)new KaiserBessel(kw.Get(), osamp.Get(), (info.type == Info::Type::ThreeD))
           : (Kernel *)new NearestNeighbour(kw ? kw.Get() : 1);
    GridOp gridder(traj.mapping(osamp.Get(), kernel->radius()), kernel, fastgrid, log);
    dc = SDC::Pipe(traj, gridder, log);
  } else if (sdc.Get() == "radial") {
    dc = SDC::Radial(traj, log);
  } else {
    Log::Fail(FMT_STRING("Uknown SDC method: {}"), sdc.Get());
  }
  HD5::Writer writer(OutName(iname.Get(), oname.Get(), "sdc", "h5"), log);
  writer.writeInfo(info);
  writer.writeSDC(dc);
  return EXIT_SUCCESS;
}
