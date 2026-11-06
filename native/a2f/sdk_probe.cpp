#include "audio2face/audio2face.h"
#include "audio2x/cuda_utils.h"
#include <iostream>

int main() {
  const auto error = nva2x::SetCudaDeviceIfNeeded(0);
  if (error) {
    std::cerr << error.message() << '\n';
    return 1;
  }
  auto* cpu = nva2f::CreateBlendshapeSolver(false);
  if (!cpu) return 2;
  cpu->Destroy();
  auto* gpu = nva2f::CreateBlendshapeSolver(true);
  if (!gpu) return 3;
  gpu->Destroy();
  std::cout << "A2F SDK CPU/GPU solver factories created; no model loaded or solved.\n";
}
