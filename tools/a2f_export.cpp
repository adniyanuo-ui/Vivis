// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.

#include "audio2face/audio2face.h"
#include "audio2x/cuda_utils.h"

#include <cstdint>
#include "AudioFile.h"

#include <cuda_runtime_api.h>
#include <any>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <iostream>
#include <memory>

//
// Boilerplate utilities for sample.
//

#define CHECK_RESULT(func)                                                     \
  {                                                                            \
    std::error_code error = (func);                                            \
    if (error) {                                                               \
      std::cout << "Error (" << __LINE__ << "): Failed to execute: " << #func; \
      std::cout << ", Reason: "<< error.message() << std::endl;                \
      exit(1);                                                                 \
    }                                                                          \
  }

#define CHECK_ERROR(expression)                                                \
  {                                                                            \
    if (!(expression)) {                                                       \
      std::cout << "Error (" << __LINE__ << "): " << #expression;              \
      std::cout << " is NULL" << std::endl;                                    \
      exit(1);                                                                 \
    }                                                                          \
  }

struct Destroyer {
  template <typename T> void operator()(T *obj) const {
    obj->Destroy();
  }
};
template <typename T> using UniquePtr = std::unique_ptr<T, Destroyer>;
template <typename T> UniquePtr<T> ToUniquePtr(T* ptr) { return UniquePtr<T>(ptr); }

std::vector<float> readAudio(std::string_view audioFilePath) {
  AudioFile<float> audioFile;
  std::cout << "Loading audio file: " << audioFilePath << std::endl;
  CHECK_ERROR(audioFile.load(audioFilePath.data()));
  audioFile.printSummary();
  CHECK_ERROR(audioFile.getSampleRate() == 16000);

  return audioFile.samples[0];
}

std::string evaAudioPath = TEST_DATA_DIR "assets/audio/official_4sec.wav";
std::string evaModelPath = TEST_DATA_DIR "assets/models/mark-v2.3/model.json";
std::string evaOutputPath = TEST_DATA_DIR "outputs/5_animation";
std::string evaEmotionPath;
using EvaClock = std::chrono::steady_clock;
double EvaSeconds(EvaClock::time_point start) {
  return std::chrono::duration<double>(EvaClock::now()-start).count();
}
nlohmann::json evaTiming;
bool evaCountOnly = false;

std::vector<float> loadAudio() {
  const std::string_view audioFilePath = evaAudioPath;
  return readAudio(audioFilePath);
}


//
// Helpers to create geometry executors.
//

// Create a regression geometry executor using a bundle, which handles the
// creation of the various objects needed for the geometry executor.
UniquePtr<nva2f::IGeometryExecutorBundle> CreateRegressionGeometryExecutorBundle() {
  const char* filename = evaModelPath.c_str();
  auto bundle = ToUniquePtr(
    nva2f::ReadRegressionGeometryExecutorBundle(
      1,
      filename,
      nva2f::IGeometryExecutor::ExecutionOption::All,
      60, 1,
      nullptr
      )
    );
  CHECK_ERROR(bundle);

  return bundle;
}

// Create a diffusion geometry executor using a bundle, which handles the
// creation of the various objects needed for the geometry executor.
UniquePtr<nva2f::IGeometryExecutorBundle> CreateDiffusionGeometryExecutorBundle() {
  const char* filename = evaModelPath.c_str();
  auto bundle = ToUniquePtr(
    nva2f::ReadDiffusionGeometryExecutorBundle(
      1,
      filename,
      nva2f::IGeometryExecutor::ExecutionOption::All,
      0,
      true,
      nullptr
      )
    );
  CHECK_ERROR(bundle);

  return bundle;
}


//
// Helper to setup emotions.
//

// Add a default emotion to the emotion accumulator.
void AddDefaultEmotion(nva2f::IGeometryExecutorBundle& bundle) {
  nlohmann::json timeline;
  if (!evaEmotionPath.empty()) {
    std::ifstream file(evaEmotionPath); file >> timeline;
    CHECK_ERROR(file.good() || file.eof());
    std::ifstream networkFile(std::filesystem::path(evaModelPath).parent_path()/"network_info.json");
    nlohmann::json network; networkFile >> network;
    CHECK_ERROR(network["id"]["type"] == "diffusion");
    CHECK_ERROR(timeline["emotion_names"] == network["params"]["emotions"]);
  }
  const auto nbTracks=bundle.GetExecutor().GetNbTracks();
  for (std::size_t track=0;track<nbTracks;++track) {
    auto& accumulator=bundle.GetEmotionAccumulator(track);
    if (evaEmotionPath.empty()) {
      std::vector<float> zero(accumulator.GetEmotionSize(),0.0f);
      CHECK_RESULT(accumulator.Accumulate(0,nva2x::HostTensorFloatConstView{zero.data(),zero.size()},bundle.GetCudaStream().Data()));
    } else {
      std::int64_t previous=-1;
      for(const auto& key:timeline["keyframes"]) {
        const auto timestamp=static_cast<std::int64_t>(std::llround(key["timestamp_ms"].get<double>()*16));
        CHECK_ERROR(timestamp>previous);previous=timestamp;
        auto weights=key["weights"].get<std::vector<float>>();
        CHECK_ERROR(weights.size()==accumulator.GetEmotionSize());
        for(float value:weights) CHECK_ERROR(std::isfinite(value) && value>=0 && value<=1);
        CHECK_RESULT(accumulator.Accumulate(timestamp,nva2x::HostTensorFloatConstView{weights.data(),weights.size()},bundle.GetCudaStream().Data()));
        CHECK_ERROR(cudaStreamSynchronize(bundle.GetCudaStream().Data())==cudaSuccess);
      }
    }
    CHECK_RESULT(accumulator.Close());
  }
}


//
// Functions to run the executors.
//

// This function runs the geometry executor offline, with the audio already accumulated.
void RunExecutorOffline(
  UniquePtr<nva2f::IGeometryExecutorBundle> geometryExecutorBundle
  ) {
  //
  // Setup
  //
  const auto nbTracks = geometryExecutorBundle->GetExecutor().GetNbTracks();

  // Set a callback on the geometry executor to get the results.
  // In this example, it simply counts the number of frames processed.
  struct GeometryExecutorCallbackData {
    std::vector<std::size_t> frameIndices;
    bool countOnly = false;
    std::ofstream skin, tongue, jaw, timestamps;
  };
  GeometryExecutorCallbackData callbackData;
  callbackData.countOnly=evaCountOnly;
  if (!evaCountOnly) {
    callbackData.skin.open(evaOutputPath + "/skin.f32",std::ios::binary);
    callbackData.tongue.open(evaOutputPath + "/tongue.f32",std::ios::binary);
    callbackData.jaw.open(evaOutputPath + "/jaw.f32",std::ios::binary);
    callbackData.timestamps.open(evaOutputPath + "/timestamps.csv");
  }
  callbackData.frameIndices.resize(nbTracks, 0);
  auto callback = [](void* userdata, const nva2f::IGeometryExecutor::Results& results) {
    auto& data = *static_cast<GeometryExecutorCallbackData*>(userdata);
    auto save = [](std::ofstream& file, nva2x::DeviceTensorFloatConstView tensor, cudaStream_t stream) {
      if (tensor.Size() == 0) return;
      std::vector<float> host(tensor.Size());
      CHECK_ERROR(cudaStreamSynchronize(stream) == cudaSuccess);
      CHECK_ERROR(cudaMemcpy(host.data(), tensor.Data(), tensor.Size()*sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
      file.write(reinterpret_cast<const char*>(host.data()), host.size()*sizeof(float));
      CHECK_ERROR(file.good());
    };
    if (!data.countOnly) {
      save(data.skin, results.skinGeometry, results.skinCudaStream);
      save(data.tongue, results.tongueGeometry, results.tongueCudaStream);
      save(data.jaw, results.jawTransform, results.jawCudaStream);
      data.timestamps << data.frameIndices[results.trackIndex] << "," << results.timeStampCurrentFrame << "\n";
    }
    data.frameIndices[results.trackIndex]++;
    return true;
  };
  CHECK_RESULT(geometryExecutorBundle->GetExecutor().SetResultsCallback(callback, &callbackData));

  // Then, load all the audio and accumulate it.
  const auto audioStart=EvaClock::now();
  const auto audioBuffer = loadAudio();
  CHECK_ERROR(!audioBuffer.empty());
  for (std::size_t trackIndex = 0; trackIndex < nbTracks; ++trackIndex) {
    // We put different amount of audio in each track to test the executor.
    CHECK_RESULT(
      geometryExecutorBundle->GetAudioAccumulator(trackIndex).Accumulate(
        nva2x::HostTensorFloatConstView{audioBuffer.data(), audioBuffer.size() / (1 + trackIndex)}, geometryExecutorBundle->GetCudaStream().Data()
        )
      );
    CHECK_RESULT(geometryExecutorBundle->GetAudioAccumulator(trackIndex).Close());
  }
  CHECK_ERROR(cudaDeviceSynchronize()==cudaSuccess);
  evaTiming["audio_load_upload_seconds"]=EvaSeconds(audioStart);
  evaTiming["audio_duration_seconds"]=audioBuffer.size()/16000.0;

  //
  // Execution.
  //

  // Process all geometry.
  const auto executeStart=EvaClock::now();
  while (nva2x::GetNbReadyTracks(geometryExecutorBundle->GetExecutor()) > 0) {
    CHECK_RESULT(geometryExecutorBundle->GetExecutor().Execute(nullptr));
  }
  CHECK_ERROR(cudaDeviceSynchronize()==cudaSuccess);
  evaTiming["execute_and_callback_seconds"]=EvaSeconds(executeStart);
  evaTiming["frames"]=callbackData.frameIndices[0];
  const auto flushStart=EvaClock::now();
  if (!evaCountOnly) {
    callbackData.skin.flush();callbackData.tongue.flush();callbackData.jaw.flush();callbackData.timestamps.flush();
    CHECK_ERROR(callbackData.skin.good() && callbackData.tongue.good() && callbackData.jaw.good() && callbackData.timestamps.good());
  }
  evaTiming["output_stream_flush_seconds"]=EvaSeconds(flushStart);

  std::size_t numerator=0,denominator=0;
  geometryExecutorBundle->GetExecutor().GetFrameRate(numerator,denominator);
  std::ofstream metadata(evaOutputPath + "/export_info.json");
  metadata << "{\"frames\":" << callbackData.frameIndices[0]
           << ",\"skin_vertices\":" << geometryExecutorBundle->GetExecutor().GetSkinGeometrySize()/3
           << ",\"tongue_vertices\":" << geometryExecutorBundle->GetExecutor().GetTongueGeometrySize()/3
           << ",\"fps_numerator\":" << numerator << ",\"fps_denominator\":" << denominator
           << ",\"constant_noise\":true,\"identity_index\":0}\n";
  CHECK_ERROR(metadata.good());
  for (std::size_t trackIndex = 0; trackIndex < nbTracks; ++trackIndex) {
    std::cout << "Track " << trackIndex << " processed " << callbackData.frameIndices[trackIndex] << " frames." << std::endl;
  }
}


//
// Combinations of previous setup functions for complete workflows.
//

// This function creates a regression geometry executor using a bundle,
// and runs the executor offline, with the audio already accumulated.
void RunRegressionBundleOffline() {
  const auto modelStart=EvaClock::now();
  auto geometryExecutorData = CreateRegressionGeometryExecutorBundle();
  CHECK_ERROR(cudaDeviceSynchronize()==cudaSuccess);
  evaTiming["model_load_seconds"]=EvaSeconds(modelStart);
  const auto emotionStart=EvaClock::now();
  AddDefaultEmotion(*geometryExecutorData);
  CHECK_ERROR(cudaDeviceSynchronize()==cudaSuccess);
  evaTiming["emotion_setup_seconds"]=EvaSeconds(emotionStart);
  RunExecutorOffline(std::move(geometryExecutorData));
}

// This function creates a diffusion geometry executor using a bundle,
// and runs the executor offline, with the audio already accumulated.
void RunDiffusionBundleOffline() {
  const auto modelStart=EvaClock::now();
  auto geometryExecutorData = CreateDiffusionGeometryExecutorBundle();
  CHECK_ERROR(cudaDeviceSynchronize()==cudaSuccess);
  evaTiming["model_load_seconds"]=EvaSeconds(modelStart);
  const auto emotionStart=EvaClock::now();
  AddDefaultEmotion(*geometryExecutorData);
  CHECK_ERROR(cudaDeviceSynchronize()==cudaSuccess);
  evaTiming["emotion_setup_seconds"]=EvaSeconds(emotionStart);
  RunExecutorOffline(std::move(geometryExecutorData));
}


int main(int argc, char** argv) {
  const auto processStart=EvaClock::now();
  evaCountOnly=std::getenv("EVA_A2F_COUNT_ONLY") && std::string(std::getenv("EVA_A2F_COUNT_ONLY"))=="1";
  if (argc > 1) evaAudioPath = argv[1];
  if (argc > 2) evaModelPath = argv[2];
  if (argc > 3) evaOutputPath = argv[3];
  if (argc > 5) evaEmotionPath = argv[5];
  const bool diffusion = argc > 4 && std::string(argv[4]) == "diffusion";
  std::filesystem::create_directories(evaOutputPath);
  std::cout << "================================================" << std::endl;
  std::cout << "    Audio2Face SDK Example" << std::endl;
  std::cout << "    Audio2Face Multi-Track Executor" << std::endl;
  std::cout << "================================================" << std::endl;

  constexpr int deviceID = 0;
  const auto cudaStart=EvaClock::now();
  CHECK_RESULT(nva2x::SetCudaDeviceIfNeeded(deviceID));
  CHECK_ERROR(cudaDeviceSynchronize()==cudaSuccess);
  evaTiming["cuda_context_initialization_seconds"]=EvaSeconds(cudaStart);

  std::cout << "\nRunning geometry bundle offline...\n" << std::endl;
  if (diffusion) RunDiffusionBundleOffline(); else RunRegressionBundleOffline();
  std::cout << "\nDone.\n" << std::endl;



  std::cout << "================================================" << std::endl;
  std::cout << "    Finished" << std::endl;
  std::cout << "================================================" << std::endl;

  evaTiming["in_process_total_seconds"]=EvaSeconds(processStart);
  evaTiming["mode"]=evaCountOnly ? "inference_with_count_callback_no_geometry_export" : "inference_with_cpu_geometry_copy_and_file_export";
  evaTiming["synchronization"]="cudaDeviceSynchronize before and after timed execution";
  evaTiming["notes"]="Wall-clock SDK execution, not isolated GPU kernel time; prebuilt TensorRT engine; file writes use OS cache, no fsync";
  evaTiming["model_path"]=evaModelPath;
  std::ofstream timing(evaOutputPath+"/timing.json");
  timing<<evaTiming.dump(2)<<"\n";
  CHECK_ERROR(timing.good());

  return 0;
}
