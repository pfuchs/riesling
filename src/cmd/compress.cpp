#include "algo/decomp.hpp"
#include "algo/stats.hpp"
#include "compressor.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "parse_args.hpp"
#include "sense/rovir.hpp"
#include "tensorOps.hpp"
#include "types.hpp"

#include <Eigen/Eigenvalues>

using namespace rl;

int main_compress(args::Subparser &parser)
{
  CoreOpts coreOpts(parser);

  // One of the following must be set
  args::Flag                   pca(parser, "V", "Calculate PCA compression", {"pca"});
  args::Flag                   rovir(parser, "R", "Calculate ROVIR compression", {"rovir"});
  args::ValueFlag<std::string> ccFile(parser, "F", "Read compression matrix from file", {"cc-file"});

  // General options
  args::Flag             save(parser, "S", "Save compression matrix to .h5 file", {"save"});
  args::ValueFlag<Index> nRetain(parser, "C", "Retain N channels (8)", {"channels"}, 8);
  args::ValueFlag<float> energy(parser, "E", "Retain fraction energy (overrides channels)", {"energy"}, -1.f);
  args::ValueFlag<Index> refVol(parser, "V", "Use this volume (default first)", {"vol"}, 0);
  args::ValueFlag<Index> lores(parser, "L", "Number of lores traces", {"lores"}, 0);

  // PCA Options
  args::ValueFlag<Sz2, SzReader<2>> pcaRead(parser, "R", "PCA Samples (start, size)", {"pca-samp"}, Sz2{0, 16});
  args::ValueFlag<Sz3, SzReader<3>> pcaTraces(parser, "R", "PCA Traces (start, size, stride)", {"pca-traces"}, Sz3{0, 1024, 4});
  args::ValueFlag<Sz2, SzReader<2>> pcaSlices(parser, "R", "PCA Slices (start, size)", {"pca-slices"}, Sz2{0, 1});
  ROVIROpts                       rovirOpts(parser);

  ParseCommand(parser, coreOpts.iname);

  HD5::Reader      reader(coreOpts.iname.Get());
  Trajectory const traj(reader.readInfo(), reader.readTensor<Re3>(HD5::Keys::Trajectory));
  Cx4 const        ks = reader.readSlab<Cx4>(HD5::Keys::Noncartesian, {refVol.Get()}, {0});
  Index const      channels = ks.dimension(0);
  Index const      samples = ks.dimension(1);
  Index const      traces = ks.dimension(2);
  Eigen::MatrixXcf psi;
  if (pca) {
    Index const maxRead = samples - pcaRead.Get()[0];
    Index const nread = (pcaRead.Get()[1] > maxRead) ? maxRead : pcaRead.Get()[1];
    if (pcaTraces.Get()[0] + pcaTraces.Get()[1] > traces) {
      Log::Fail("Requested end spoke {} is past end of file {}", pcaTraces.Get()[0] + pcaTraces.Get()[1], traces);
    }
    Log::Print("Using {} read points, {} traces, {} stride", nread, pcaTraces.Get()[1], pcaTraces.Get()[2]);
    Cx4 const ref = ks.slice(Sz4{0, pcaRead.Get()[0], pcaTraces.Get()[0], pcaSlices.Get()[0]},
                             Sz4{channels, pcaRead.Get()[1], pcaTraces.Get()[1], pcaSlices.Get()[1]})
                      .stride(Sz4{1, 1, pcaTraces.Get()[2], 1});
    auto const cov = Covariance(CollapseToConstMatrix(ref));
    auto const eig = Eig<Cx>(cov);
    auto const nR = energy ? Threshold(eig.V, energy.Get()) : nRetain.Get();
    psi = eig.P.leftCols(nR);
  } else if (rovir) {
    psi = ROVIR(rovirOpts, traj, energy.Get(), nRetain.Get(), lores.Get(), ks);
  } else if (ccFile) {
    HD5::Reader matFile(ccFile.Get());
    psi = matFile.readMatrix<Eigen::MatrixXcf>(HD5::Keys::CompressionMatrix);
  } else {
    Log::Fail("Must specify PCA/ROVIR/load from file");
  }
  Compressor  compressor{psi};
  Cx5 const   uncompressed = reader.readTensor<Cx5>(HD5::Keys::Noncartesian);
  Cx5         compressed(AddFront(LastN<4>(uncompressed.dimensions()), psi.cols()));
  Index const volumes = uncompressed.dimension(4);
  for (Index iv = 0; iv < volumes; iv++) {
    compressed.chip<4>(iv) = compressor.compress(Cx4(uncompressed.chip<4>(iv)));
  }

  HD5::Writer writer(OutName(coreOpts.iname.Get(), coreOpts.oname.Get(), parser.GetCommand().Name()));
  writer.writeInfo(traj.info());
  writer.writeTensor(HD5::Keys::Trajectory, traj.points().dimensions(), traj.points().data());
  writer.writeTensor(HD5::Keys::Noncartesian, compressed.dimensions(), compressed.data());

  if (save) {
    HD5::Writer matfile(OutName(coreOpts.iname.Get(), coreOpts.oname.Get(), "ccmat"));
    matfile.writeMatrix(compressor.psi, HD5::Keys::CompressionMatrix);
  }
  return EXIT_SUCCESS;
}
