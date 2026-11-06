// Validate the platform adapter's reconstruction and floating-mode restoration.
#include "tbtSVD/SVD.h"
#include <cmath>
#include <iostream>

int main() {
  double maximum = 0;
  for(int i=0;i<64;++i) {
    SVD::Mat3x3 a;
    auto* data = &a.m_00;
    for(int j=0;j<9;++j) data[j]=std::sin(float(i*9+j+1)*.37f);
    a.m_00+=2; a.m_11+=3; a.m_22+=4;
    unsigned long before=0,after=0;
#if defined(__aarch64__)
    asm volatile("mrs %0, fpcr" : "=r"(before));
#endif
    const auto decomposition=SVD::svd(a);
#if defined(__aarch64__)
    asm volatile("mrs %0, fpcr" : "=r"(after));
#endif
    if(before!=after) return 2;
    const auto restored=decomposition.U*decomposition.S*decomposition.V.transpose();
    const auto* actual=&restored.m_00;
    double squared=0,norm=0;
    for(int j=0;j<9;++j) { squared+=std::pow(actual[j]-data[j],2); norm+=data[j]*data[j]; }
    const double error=std::sqrt(squared/norm);
    if(!std::isfinite(error) || error>.001) return 3;
    maximum=std::max(maximum,error);
  }
  std::cout<<"64 SVD reconstructions passed; max relative error="<<maximum<<"; FPCR restored\n";
}
