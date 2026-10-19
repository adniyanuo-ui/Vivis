#include "audio2face/audio2face.h"
#include "audio2x/cuda_utils.h"
#include <cuda_runtime_api.h>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
using json=nlohmann::json;
struct Destroy { template<class T> void operator()(T* p)const{if(p)p->Destroy();} };
template<class T> using Ptr=std::unique_ptr<T,Destroy>;
void check(std::error_code e){if(e)throw std::runtime_error(e.message());}
void cuda_check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
int main(int argc,char**argv){try{
 if(argc!=4)throw std::runtime_error("Usage: eva-a2f-solve MODEL_FOLDER ANIMATION_FOLDER regression|diffusion");
 const std::filesystem::path model=argv[1],folder=argv[2];
 const std::string suffix=std::string(argv[3])=="diffusion"?"_Claire":"";
 check(nva2x::SetCudaDeviceIfNeeded(0));
 std::ifstream metadata_file(folder/"export_info.json");json info;metadata_file>>info;
 const size_t frames=info.at("frames");json bindings;
 for(const std::string part:{"skin","tongue"}){
  auto config=Ptr<nva2f::IBlendshapeSolverConfig>(nva2f::ReadBlendshapeSolverConfig((model/("bs_"+part+"_config"+suffix+".json")).c_str()));
  auto data=Ptr<nva2f::IBlendshapeSolverData>(nva2f::ReadBlendshapeSolverData((model/("bs_"+part+suffix+".npz")).c_str()));
  auto solver=Ptr<nva2f::IBlendshapeSolver>(nva2f::CreateBlendshapeSolver(true));
  if(!config||!data||!solver)throw std::runtime_error("Cannot create official blendshape solver");
  check(solver->SetBlendshapeData(data->GetBlendshapeSolverDataView()));
  check(solver->SetBlendshapeConfig(config->GetBlendshapeSolverConfig()));
  check(solver->SetParameters(config->GetBlendshapeSolverParams()));check(solver->Prepare());
  const size_t floats=data->GetBlendshapeSolverDataView().neutralPose.Size();
  const size_t count=solver->NumBlendshapePoses();std::vector<float> target(floats),weights(count);
  std::vector<int> active(count);check(solver->GetActivePoses(active.data(),active.size()));
  std::vector<std::string> names;for(size_t i=0;i<count;++i)names.push_back(solver->GetPoseName(i));
  bindings[part]={{"names",names},{"active",active},{"vertices",floats/3},{"count",count}};
  std::ifstream input(folder/(part+".f32"),std::ios::binary);
  std::ofstream output(folder/("weights_"+part+".f32"),std::ios::binary);
  auto device=Ptr<nva2x::IDeviceTensorFloat>(nva2x::CreateDeviceTensorFloat(floats));
  if(!device)throw std::runtime_error("Cannot allocate device tensor");
  for(size_t f=0;f<frames;++f){
   input.read(reinterpret_cast<char*>(target.data()),floats*sizeof(float));if(!input)throw std::runtime_error("Truncated geometry input");
   cuda_check(cudaMemcpy(device->Data(),target.data(),floats*sizeof(float),cudaMemcpyHostToDevice));
   check(solver->Solve(device->View(0,floats),{weights.data(),weights.size()}));
   output.write(reinterpret_cast<const char*>(weights.data()),weights.size()*sizeof(float));
   if(!output)throw std::runtime_error("Cannot write weights");
  }
  std::cout<<part<<": "<<frames<<" frames, "<<count<<" named weights\n";
 }
 std::ofstream result(folder/"source_bindings.json");result<<bindings.dump(2)<<"\n";
 return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<"\n";return 1;}}
