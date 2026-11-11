// Regress ROI brightness determinism without photos, cameras or model loading.
#include <seeta/QualityOfBrightness.h>
#include <cmath>
#include <iostream>
#include <vector>
int main() {
    seeta::QualityOfBrightness quality;
    SeetaRect face{100,100,80,100};
    for (int face_level : {150,20}) {
        std::vector<unsigned char> pixels(300*300*3,245);
        for (int y=100;y<200;++y)
            for (int x=100;x<180;++x)
                for (int c=0;c<3;++c) pixels[(y*300+x)*3+c]=face_level;
        const auto original=pixels;
        SeetaImageData image{300,300,3,pixels.data()};
        float previous=-1;
        for (int i=0;i<50;++i) {
            auto result=quality.check(image,face,nullptr,0);
            if (!std::isfinite(result.score) || result.level!=(face_level==150?seeta::HIGH:seeta::LOW)
                || (i && result.score!=previous) || pixels!=original) {
                std::cerr << "ROI brightness/determinism/input immutability failed\n";return 1;
            }
            previous=result.score;
        }
    }
    std::cout << "100 synthetic ROI brightness checks passed\n";
}
