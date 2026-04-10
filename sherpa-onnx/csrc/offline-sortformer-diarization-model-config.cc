// sherpa-onnx/csrc/offline-sortformer-diarization-model-config.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-sortformer-diarization-model-config.h"

#include <sstream>
#include <string>

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

void OfflineSortformerDiarizationModelConfig::Register(ParseOptions *po) {
  po->Register("sortformer-model", &model,
               "Path to the NVIDIA Sortformer streaming diarization ONNX "
               "model, e.g. diar_streaming_sortformer_4spk-v2.onnx");

  po->Register("sortformer-num-threads", &num_threads,
               "Number of threads to run the Sortformer neural network");

  po->Register("sortformer-debug", &debug,
               "true to print model information while loading it.");

  po->Register("sortformer-provider", &provider,
               "Execution provider to use: cpu, cuda, coreml");
}

bool OfflineSortformerDiarizationModelConfig::Validate() const {
  if (num_threads < 1) {
    SHERPA_ONNX_LOGE("num_threads should be > 0. Given %d", num_threads);
    return false;
  }

  if (model.empty()) {
    SHERPA_ONNX_LOGE("Please provide --sortformer-model");
    return false;
  }

  if (!FileExists(model)) {
    SHERPA_ONNX_LOGE("Sortformer model '%s' does not exist", model.c_str());
    return false;
  }

  return true;
}

std::string OfflineSortformerDiarizationModelConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineSortformerDiarizationModelConfig(";
  os << "model=\"" << model << "\", ";
  os << "num_threads=" << num_threads << ", ";
  os << "debug=" << (debug ? "True" : "False") << ", ";
  os << "provider=\"" << provider << "\")";

  return os.str();
}

}  // namespace sherpa_onnx
