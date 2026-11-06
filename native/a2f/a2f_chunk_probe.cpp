// Isolated A2F deployment diagnostic. No camera, serial or audio-output code.
#include "audio2face/audio2face.h"
#include "audio2x/cuda_utils.h"
#include "AudioFile.h"
#include <cuda_runtime_api.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

using json = nlohmann::json;
using Clock = std::chrono::steady_clock;
struct Destroy { template<class T> void operator()(T* p) const { if(p) p->Destroy(); } };
template<class T> using Owner = std::unique_ptr<T, Destroy>;
void check(std::error_code error) { if(error) throw std::runtime_error(error.message()); }
void cuda_check(cudaError_t e) { if(e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
double elapsed(Clock::time_point start) { return std::chrono::duration<double>(Clock::now()-start).count(); }
struct Capture {
  std::ofstream frames;
  bool streaming = false;
  std::string playback_id;
  Clock::time_point start;
  size_t count = 0;
  double first = -1;
  long long last_sample = -1;
  size_t invalid = 0;
};

int main(int argc, char** argv) { try {
  if(argc != 5 && argc != 7) throw std::runtime_error("Usage: a2f_chunk_probe MODEL_JSON WAV NEW_OUTPUT_DIR SECONDS [--stream PLAYBACK_ID]");
  const bool streaming = argc==7 && std::string(argv[5])=="--stream";
  if(argc==7 && !streaming) throw std::runtime_error("Unknown option");
  const std::filesystem::path out(argv[3]);
  if(std::filesystem::exists(out)) throw std::runtime_error("Output directory already exists");
  const double seconds = std::stod(argv[4]);
  if(!std::isfinite(seconds) || seconds <= 0 || seconds > 3600) throw std::runtime_error("Invalid duration");
  AudioFile<float> wav;
  if(!wav.load(argv[2]) || wav.getSampleRate()!=16000 || wav.getNumChannels()!=1)
    throw std::runtime_error("Need 16kHz mono WAV");
  const size_t samples = std::min(wav.samples[0].size(), static_cast<size_t>(std::llround(seconds*16000)));
  if(!samples) throw std::runtime_error("Empty audio");
  cuda_check(cudaSetDevice(0));
  auto load = Clock::now();
  Owner<nva2f::IBlendshapeExecutorBundle> bundle(nva2f::ReadDiffusionBlendshapeSolveExecutorBundle(
    1, argv[1], nva2f::IGeometryExecutor::ExecutionOption::SkinTongue,
    true, 0, true, nullptr, nullptr));
  if(!bundle) throw std::runtime_error("Cannot load diffusion bundle");
  json report = {{"scope","unpaced 250ms audio chunk deployment test; not DAC playback or robot acceptance"},
                 {"model_load_seconds",elapsed(load)}, {"audio_samples",samples},
                 {"audio_duration_seconds",samples/16000.}, {"sample_rate_hz",16000}};
  auto& executor = bundle->GetExecutor();
  report["weights_count"] = executor.GetWeightCount();
  for(const std::string part : {"skin","tongue"}) {
    nva2f::IBlendshapeSolver* solver = nullptr;
    if(part=="skin") check(nva2f::GetExecutorSkinSolver(executor, 0, &solver));
    else check(nva2f::GetExecutorTongueSolver(executor, 0, &solver));
    if(!solver) throw std::runtime_error("Missing solver");
    report["names"][part] = json::array();
    for(int i=0;i<solver->NumBlendshapePoses();++i) report["names"][part].push_back(solver->GetPoseName(i));
  }
  std::filesystem::create_directories(out);
  Capture capture;
  capture.streaming=streaming;
  if(streaming) {
    capture.playback_id=argv[6];
    std::cout << json({{"schema_version","vivis-a2f-bindings-1.0"},
                      {"playback_id",capture.playback_id},{"names",report["names"]}}).dump() << '\n' << std::flush;
  } else {
    capture.frames.open(out/"frames.ndjson");
    if(!capture.frames) throw std::runtime_error("Cannot write frames");
  }
  if(executor.GetResultType()!=nva2f::IBlendshapeExecutor::ResultsType::DEVICE)
    throw std::runtime_error("Unexpected solver result interface");
  check(executor.SetResultsCallback(+[](void* data, const nva2f::IBlendshapeExecutor::DeviceResults& r) {
    auto& c = *static_cast<Capture*>(data);
    std::vector<float> weights(r.weights.Size());
    cuda_check(cudaStreamSynchronize(r.cudaStream));
    cuda_check(cudaMemcpy(weights.data(),r.weights.Data(),weights.size()*sizeof(float),cudaMemcpyDeviceToHost));
    for(float v:weights) if(!std::isfinite(v)) ++c.invalid;
    if(c.count && r.timeStampCurrentFrame<=c.last_sample) throw std::runtime_error("Non-monotonic animation timestamp");
    c.last_sample=r.timeStampCurrentFrame;
    const auto ready=(!c.streaming || !c.count) ? elapsed(c.start) : 0.;
    if(!c.count) c.first=ready;
    json frame={{"frame_seq",c.count},{"audio_sample_index",r.timeStampCurrentFrame},
                {"sample_rate_hz",16000},{"weights",weights}};
    if(c.streaming) {
      frame["schema_version"]="vivis-a2f-raw-1.0";
      frame["playback_id"]=c.playback_id;
      std::cout << frame.dump() << '\n' << std::flush;
      if(!std::cout) return false;
    } else {
      frame["ready_seconds"]=ready;
      c.frames << frame.dump() << '\n';
    }
    ++c.count;
    return true;
  }, &capture));
  auto& emotion = bundle->GetEmotionAccumulator(0);
  std::vector<float> neutral(emotion.GetEmotionSize(),0.f);
  check(emotion.Accumulate(0,nva2x::HostTensorFloatConstView{neutral.data(),neutral.size()},bundle->GetCudaStream().Data()));
  check(emotion.Close());
  cuda_check(cudaDeviceSynchronize());
  capture.start = Clock::now();
  size_t cursor = 0;
  report["chunks"] = json::array();
  while(cursor<samples) {
    const auto begin = Clock::now();
    const auto count = std::min(size_t(4000),samples-cursor);
    check(bundle->GetAudioAccumulator(0).Accumulate(nva2x::HostTensorFloatConstView{wav.samples[0].data()+cursor,count},bundle->GetCudaStream().Data()));
    cursor+=count;
    while(nva2x::GetNbReadyTracks(executor)) check(executor.Execute(nullptr));
    cuda_check(cudaDeviceSynchronize());
    if(!streaming) {
      report["chunks"].push_back({{"input_samples",cursor},{"output_frames",capture.count},{"seconds",elapsed(begin)}});
      std::cout << "input=" << cursor << " frames=" << capture.count << '\n' << std::flush;
    }
  }
  report["frames_before_audio_close"] = capture.count;
  check(bundle->GetAudioAccumulator(0).Close());
  while(nva2x::GetNbReadyTracks(executor)) check(executor.Execute(nullptr));
  check(executor.Wait(0));
  cuda_check(cudaDeviceSynchronize());
  report["inference_and_solve_seconds"] = elapsed(capture.start);
  report["real_time_factor"] = elapsed(capture.start)/(samples/16000.);
  report["first_frame_seconds"] = capture.first;
  report["frames"] = capture.count;
  report["invalid_weights"] = capture.invalid;
  report["last_audio_sample_index"] = capture.last_sample;
  report["completed"] = capture.count>0 && capture.invalid==0;
  if(!streaming) capture.frames.close();
  std::ofstream(out/"report.json") << report.dump(2) << '\n';
  (streaming ? std::cerr : std::cout) << report.dump(2) << '\n';
  return report["completed"].get<bool>() ? 0 : 2;
} catch(const std::exception& e) {
  std::cerr << e.what() << '\n';
  return 1;
}}
