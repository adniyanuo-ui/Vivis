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

#include <any>
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
  constexpr char filename[] = TEST_DATA_DIR "assets/models/mark-v2.3/model.json";
  auto bundle = ToUniquePtr(
    nva2f::ReadRegressionGeometryExecutorBundle(
      8,
      filename,
      nva2f::IGeometryExecutor::ExecutionOption::SkinTongue,
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
  constexpr char filename[] = TEST_DATA_DIR "_data/generated/audio2face-sdk/samples/data/multi-diffusion/model.json";
  auto bundle = ToUniquePtr(
    nva2f::ReadDiffusionGeometryExecutorBundle(
      8,
      filename,
      nva2f::IGeometryExecutor::ExecutionOption::SkinTongue,
      0,
      false,
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
  const auto nbTracks = bundle.GetExecutor().GetNbTracks();
  std::vector<float> emptyEmotion;
  for (std::size_t trackIndex = 0; trackIndex < nbTracks; ++trackIndex) {
    auto& emotionAccumulator = bundle.GetEmotionAccumulator(trackIndex);
    emptyEmotion.resize(emotionAccumulator.GetEmotionSize(), 0.0f);
    CHECK_RESULT(emotionAccumulator.Accumulate(
      0, nva2x::HostTensorFloatConstView{emptyEmotion.data(), emptyEmotion.size()}, bundle.GetCudaStream().Data()
      ));
    CHECK_RESULT(emotionAccumulator.Close());
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
  };
  GeometryExecutorCallbackData callbackData;
  callbackData.frameIndices.resize(nbTracks, 0);
  auto callback = [](void* userdata, const nva2f::IGeometryExecutor::Results& results) {
    auto& data = *static_cast<GeometryExecutorCallbackData*>(userdata);
    data.frameIndices[results.trackIndex]++;
    return true;
  };
  CHECK_RESULT(geometryExecutorBundle->GetExecutor().SetResultsCallback(callback, &callbackData));

  // Then, load all the audio and accumulate it.
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

  //
  // Execution.
  //

  // Process all geometry.
  while (nva2x::GetNbReadyTracks(geometryExecutorBundle->GetExecutor()) > 0) {
    CHECK_RESULT(geometryExecutorBundle->GetExecutor().Execute(nullptr));
  }

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
  auto geometryExecutorData = CreateRegressionGeometryExecutorBundle();
  AddDefaultEmotion(*geometryExecutorData);
  RunExecutorOffline(std::move(geometryExecutorData));
}

// This function creates a diffusion geometry executor using a bundle,
// and runs the executor offline, with the audio already accumulated.
void RunDiffusionBundleOffline() {
  auto geometryExecutorData = CreateDiffusionGeometryExecutorBundle();
  AddDefaultEmotion(*geometryExecutorData);
  RunExecutorOffline(std::move(geometryExecutorData));
}


int main(int argc, char** argv) {
  if (argc > 1) evaAudioPath = argv[1];
  std::cout << "================================================" << std::endl;
  std::cout << "    Audio2Face SDK Example" << std::endl;
  std::cout << "    Audio2Face Multi-Track Executor" << std::endl;
  std::cout << "================================================" << std::endl;

  constexpr int deviceID = 0;
  CHECK_RESULT(nva2x::SetCudaDeviceIfNeeded(deviceID));

  std::cout << "\nRunning regression bundle offline...\n" << std::endl;
  RunRegressionBundleOffline();
  std::cout << "\nDone.\n" << std::endl;



  std::cout << "================================================" << std::endl;
  std::cout << "    Finished" << std::endl;
  std::cout << "================================================" << std::endl;

  return 0;
}
