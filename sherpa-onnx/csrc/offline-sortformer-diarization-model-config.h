// sherpa-onnx/csrc/offline-sortformer-diarization-model-config.h
//
// Copyright (c)  2026  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_SORTFORMER_DIARIZATION_MODEL_CONFIG_H_
#define SHERPA_ONNX_CSRC_OFFLINE_SORTFORMER_DIARIZATION_MODEL_CONFIG_H_

#include <string>

#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct OfflineSortformerDiarizationModelConfig {
  // Path to the exported NVIDIA Sortformer streaming ONNX model, e.g.
  // diar_streaming_sortformer_4spk-v2.onnx
  std::string model;

  int32_t num_threads = 1;
  bool debug = false;
  std::string provider = "cpu";

  OfflineSortformerDiarizationModelConfig() = default;

  OfflineSortformerDiarizationModelConfig(const std::string &model,
                                          int32_t num_threads, bool debug,
                                          const std::string &provider)
      : model(model),
        num_threads(num_threads),
        debug(debug),
        provider(provider) {}

  void Register(ParseOptions *po);

  bool Validate() const;

  std::string ToString() const;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_SORTFORMER_DIARIZATION_MODEL_CONFIG_H_
