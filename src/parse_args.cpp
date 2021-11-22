#include "parse_args.h"
#include "io.h"
#include "tensorOps.h"
#include "threads.h"
#include <algorithm>
#include <filesystem>
#include <fmt/format.h>
#include <scn/scn.h>

namespace {
std::unordered_map<int, Log::Level> levelMap{
  {0, Log::Level::None}, {1, Log::Level::Info}, {2, Log::Level::Debug}, {3, Log::Level::Images}};
}

void Vector3fReader::operator()(
  std::string const &name, std::string const &value, Eigen::Vector3f &v)
{
  float x, y, z;
  auto result = scn::scan(value, "{},{},{}", x, y, z);
  if (!result) {
    Log::Fail("Could not read vector for {} from value {} because {}", name, value, result.error());
  }
  v.x() = x;
  v.y() = y;
  v.z() = z;
}

void VectorReader::operator()(
  std::string const &name, std::string const &input, std::vector<float> &values)
{
  float val;
  auto result = scn::scan(input, "{}", val);
  if (result) {
    values.push_back(val);
  } else {
    Log::Fail("Could not read argument for {}", name);
  }
  while ((result = scn::scan(result.range(), ",{}", val))) {
    values.push_back(val);
  }
}

args::Group global_group("GLOBAL OPTIONS");
args::HelpFlag help(global_group, "HELP", "Show this help message", {'h', "help"});
args::Flag verbose(global_group, "VERBOSE", "Talk more", {'v', "verbose"});
args::MapFlag<int, Log::Level> verbosity(
  global_group,
  "VERBOSITY",
  "Talk even more (values 0-3, see documentation)",
  {"verbosity"},
  levelMap);
args::ValueFlag<long> nthreads(global_group, "THREADS", "Limit number of threads", {"nthreads"});

Log ParseCommand(args::Subparser &parser, args::Positional<std::string> &iname)
{
  parser.Parse();
  Log::Level const level =
    verbosity ? verbosity.Get() : (verbose ? Log::Level::Info : Log::Level::None);

  Log log(level);
  log.info(FMT_STRING("Starting: {}"), parser.GetCommand().Name());
  if (!iname) {
    throw args::Error("No input file specified");
  }
  if (nthreads) {
    log.info("Using {} threads", nthreads.Get());
    Threads::SetGlobalThreadCount(nthreads.Get());
  }
  return log;
}

Log ParseCommand(args::Subparser &parser)
{
  parser.Parse();
  Log::Level const level =
    verbosity ? verbosity.Get() : (verbose ? Log::Level::Info : Log::Level::None);

  Log log(level);
  if (nthreads) {
    Threads::SetGlobalThreadCount(nthreads.Get());
  }
  log.info(FMT_STRING("Starting operation: {}"), parser.GetCommand().Name());
  return log;
}

std::string OutName(
  std::string const &iName,
  std::string const &oName,
  std::string const &suffix,
  std::string const &extension)
{
  return fmt::format(
    FMT_STRING("{}-{}.{}"),
    oName.empty() ? std::filesystem::path(iName).filename().replace_extension().string() : oName,
    suffix,
    extension);
}

void WriteOutput(
  Cx4 const &vols,
  bool const mag,
  bool const needsSwap,
  Info const &info,
  std::string const &iname,
  std::string const &oname,
  std::string const &suffix,
  std::string const &ext,
  Log &log)
{
  auto const fname = OutName(iname, oname, suffix, ext);
  if (ext.compare("h5") == 0) {
    HD5::Writer writer(fname, log);
    writer.writeInfo(info);
    writer.writeImage(vols);
  } else if (ext.compare("nii") == 0) {
    auto &output = needsSwap ? FirstToLast4(vols) : vols;
    if (mag) {
      R4 const mVols = output.abs();
      WriteNifti(info, mVols, fname, log);
    } else {
      WriteNifti(info, output, fname, log);
    }
  } else {
    Log::Fail("Unsupported output format: {}", ext);
  }
}

void WriteBasisVolumes(
  Cx5 const &vols,
  R2 const &basis,
  bool const mag,
  Info const &info,
  std::string const &iname,
  std::string const &oname,
  std::string const &suffix,
  std::string const &ext,
  Log &log)
{
  if (ext.compare("h5") == 0) {
    auto const fname = OutName(iname, oname, suffix, ext);
    HD5::Writer writer(fname, log);
    writer.writeInfo(info);
    writer.writeBasis(basis);
    writer.writeBasisImages(vols);
  } else {
    for (long iv = 0; iv < vols.dimension(4); iv++) {
      auto const fname = OutName(iname, oname, fmt::format("{}-{:02d}", suffix, iv), ext);
      Cx4 const output = FirstToLast4(vols.chip(iv, 4));
      if (mag) {
        R4 const mVols = output.abs();
        WriteNifti(info, mVols, fname, log);
      } else {
        WriteNifti(info, output, fname, log);
      }
    }
  }
}
