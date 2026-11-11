// Offline image evaluation only. No camera, network, audio or actuator I/O.
#include <seeta/FaceDetector.h>
#include <seeta/FaceLandmarker.h>
#include <seeta/FaceRecognizer.h>
#include <seeta/QualityOfBrightness.h>
#include <seeta/QualityOfClarity.h>
#include <seeta/AgePredictor.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

namespace {
using Clock = std::chrono::steady_clock;
thread_local std::string error;
double ms(Clock::time_point from) {
    return std::chrono::duration<double, std::milli>(Clock::now() - from).count();
}
struct Engine {
    seeta::FaceDetector detector;
    seeta::FaceLandmarker marker;
    seeta::FaceRecognizer recognizer;
    seeta::QualityOfBrightness brightness;
    seeta::QualityOfClarity clarity;
    std::unique_ptr<seeta::AgePredictor> age;
    std::string folder;
    int threads;
    Engine(const std::string &folder, int threads)
        : detector(seeta::ModelSetting(folder + "/face_detector.csta", seeta::ModelSetting::CPU)),
          marker(seeta::ModelSetting(folder + "/face_landmarker_pts5.csta", seeta::ModelSetting::CPU)),
          recognizer(seeta::ModelSetting(folder + "/face_recognizer.csta", seeta::ModelSetting::CPU)),folder(folder),threads(threads) {
        detector.set(seeta::FaceDetector::PROPERTY_NUMBER_THREADS, threads);
        detector.set(seeta::FaceDetector::PROPERTY_MIN_FACE_SIZE, 60);
        detector.set(seeta::FaceDetector::PROPERTY_THRESHOLD, .8);
        recognizer.set(seeta::FaceRecognizer::PROPERTY_NUMBER_THREADS, threads);
        if (marker.number() != 5) throw std::runtime_error("Expected five-point landmark model");
    }
};

void evaluate_face(Engine &engine, const SeetaImageData &image, const SeetaRect &rect,
                   int gate, float *features, double *meta, double *attributes=nullptr) {
    meta[1]=rect.x;meta[2]=rect.y;meta[3]=rect.width;meta[4]=rect.height;
    SeetaPointF points[5];
    auto stage=Clock::now();engine.marker.mark(image,rect,points);meta[11]=ms(stage);
    stage=Clock::now();
    auto brightness=engine.brightness.check(image,rect,points,5);
    auto clarity=engine.clarity.check(image,rect,points,5);
    meta[6]=brightness.score;meta[7]=brightness.level;meta[8]=clarity.score;meta[9]=clarity.level;meta[12]=ms(stage);
    if (gate && (brightness.level==seeta::LOW || clarity.level==seeta::LOW)) return;
    stage=Clock::now();
    if (!engine.recognizer.Extract(image,points,features)) throw std::runtime_error("SeetaFace extraction failed");
    meta[13]=ms(stage);meta[15]=1;
    if (attributes && engine.age) {
        int age=-1;stage=Clock::now();
        if (engine.age->PredictAgeWithCrop(image,points,age) && age>=0 && age<=120) attributes[0]=age;
        attributes[1]=ms(stage);
    }
}
}
extern "C" {
const char *sf6_error() { return error.c_str(); }
void *sf6_create(const char *folder, int threads) {
    try {
        if (!folder || threads < 1 || threads > 12) throw std::runtime_error("Invalid engine arguments");
        return new Engine(folder, threads);
    } catch (const std::exception &e) { error = e.what(); return nullptr; }
    catch (...) { error = "Unknown SeetaFace construction error"; return nullptr; }
}
void sf6_destroy(void *p) { delete static_cast<Engine *>(p); }
int sf6_dimension(void *p) { return static_cast<Engine *>(p)->recognizer.GetExtractFeatureSize(); }
float sf6_similarity(void *p, const float *a, const float *b) {
    return static_cast<Engine *>(p)->recognizer.CalculateSimilarity(a, b);
}
int sf6_enable_age(void *p) {
    try {
        if (!p) throw std::runtime_error("Invalid engine");
        auto &engine=*static_cast<Engine *>(p);
        engine.age.reset(new seeta::AgePredictor(seeta::ModelSetting(engine.folder+"/age_predictor.csta",seeta::ModelSetting::CPU)));
        engine.age->set(seeta::AgePredictor::PROPERTY_NUMBER_THREADS,engine.threads);return 0;
    } catch (const std::exception &e) {error=e.what();return -1;}
}
// Production path: keep the exact face selected by the existing fast YuNet.
// Never run a second largest-face selector that could identify someone else.
int sf6_eval_face(void *p,unsigned char *bgr,int width,int height,const double *box,
                  int gate,float *features,int capacity,double *meta,double *attributes) {
    try {
        if (!p || !bgr || !box || !features || !meta || !attributes || width<1 || height<1 || width>8192 || height>8192)
            throw std::runtime_error("Invalid selected-face input");
        auto &engine=*static_cast<Engine *>(p);
        if (capacity!=engine.recognizer.GetExtractFeatureSize()) throw std::runtime_error("Feature capacity mismatch");
        for (int i=0;i<4;++i) if (!std::isfinite(box[i])) throw std::runtime_error("Invalid face rectangle");
        int x=std::max(0,int(std::floor(box[0]))),y=std::max(0,int(std::floor(box[1])));
        int right=std::min(width,int(std::ceil(box[0]+box[2]))),bottom=std::min(height,int(std::ceil(box[1]+box[3])));
        if (box[2]<=0 || box[3]<=0 || right-x<16 || bottom-y<16) throw std::runtime_error("Face rectangle outside image");
        std::fill(meta,meta+16,0.);attributes[0]=-1;attributes[1]=0;
        auto begin=Clock::now();meta[0]=1;
        SeetaImageData image{width,height,3,bgr};SeetaRect rect{x,y,right-x,bottom-y};
        evaluate_face(engine,image,rect,gate,features,meta,attributes);meta[14]=ms(begin);return 0;
    } catch (const std::exception &e) {error=e.what();return -1;}
    catch (...) {error="Unknown selected-face inference error";return -1;}
}
// meta: count,xywh,detector score,brightness score/level,clarity score/level,
// detection ms,landmark ms,quality ms,recognition ms,total ms,extracted.
int sf6_eval(void *p, unsigned char *bgr, int width, int height, int quality_gate,
             float *features, int capacity, double *meta) {
    try {
        if (!p || !bgr || !features || !meta || width < 1 || height < 1 || width > 8192 || height > 8192)
            throw std::runtime_error("Invalid image buffer");
        auto &engine = *static_cast<Engine *>(p);
        if (capacity != engine.recognizer.GetExtractFeatureSize()) throw std::runtime_error("Feature capacity mismatch");
        std::fill(meta, meta + 16, 0.);
        auto begin = Clock::now();
        SeetaImageData image{width, height, 3, bgr};
        auto stage = Clock::now();
        auto faces = engine.detector.detect(image);
        meta[10] = ms(stage); meta[0] = faces.size;
        if (!faces.size) { meta[14] = ms(begin); return 0; }
        auto selected = faces.data;
        for (int i = 1; i < faces.size; ++i)
            if (faces.data[i].pos.width * faces.data[i].pos.height > selected->pos.width * selected->pos.height)
                selected = faces.data + i;
        auto rect = selected->pos;
        meta[1] = rect.x; meta[2] = rect.y; meta[3] = rect.width; meta[4] = rect.height; meta[5] = selected->score;
        evaluate_face(engine,image,rect,quality_gate,features,meta);meta[14]=ms(begin);
        return 0;
    } catch (const std::exception &e) { error = e.what(); return -1; }
    catch (...) { error = "Unknown SeetaFace inference error"; return -1; }
}
}
