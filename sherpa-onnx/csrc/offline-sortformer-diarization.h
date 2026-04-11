// sherpa-onnx/csrc/offline-sortformer-diarization.h
//
// Copyright (c)  2026  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_SORTFORMER_DIARIZATION_H_
#define SHERPA_ONNX_CSRC_OFFLINE_SORTFORMER_DIARIZATION_H_

#include <memory>
#include <string>

#include "sherpa-onnx/csrc/offline-sortformer-diarization-model-config.h"
#include "sherpa-onnx/csrc/offline-speaker-diarization-result.h"
#include "sherpa-onnx/csrc/parse-options.h"

namespace sherpa_onnx {

struct OfflineSortformerDiarizationConfig {
  OfflineSortformerDiarizationModelConfig model;

  // Probability threshold to START a speaker segment.
  float onset = 0.641f;

  // Probability threshold to END a speaker segment.
  // Must be <= onset. If negative, it defaults to onset.
  float offset = 0.561f;

  // Extra padding (in seconds) added before and after each detected speech
  // segment. Matches NeMo's VAD-style post-processing pad_onset / pad_offset.
  float pad_onset = 0.229f;
  float pad_offset = 0.079f;

  // Segments shorter than this duration (in seconds) are discarded.
  float min_duration_on = 0.511f;

  // Gaps shorter than this duration (in seconds) between two segments of the
  // same speaker cause the segments to be merged.
  float min_duration_off = 0.296f;

  // Per-speaker median filter window (in model frames, 80 ms each) applied
  // to the sigmoid predictions before binarization. Matches NeMo's default
  // median_window=11 for the callhome post-processing config. Set to 1 (or 0)
  // to disable.
  int32_t median_window = 11;

  OfflineSortformerDiarizationConfig() = default;

  OfflineSortformerDiarizationConfig(
      const OfflineSortformerDiarizationModelConfig &model, float onset,
      float offset, float pad_onset, float pad_offset, float min_duration_on,
      float min_duration_off, int32_t median_window)
      : model(model),
        onset(onset),
        offset(offset),
        pad_onset(pad_onset),
        pad_offset(pad_offset),
        min_duration_on(min_duration_on),
        min_duration_off(min_duration_off),
        median_window(median_window) {}

  void Register(ParseOptions *po);
  bool Validate() const;
  std::string ToString() const;
};

class OfflineSortformerDiarization {
 public:
  explicit OfflineSortformerDiarization(
      const OfflineSortformerDiarizationConfig &config);

  ~OfflineSortformerDiarization();

  // Expected sample rate of the input audio samples.
  int32_t SampleRate() const;

  // Number of speakers the model supports (always 4 for the public Sortformer
  // v2 model).
  int32_t NumSpeakers() const;

  // Update the post-processing thresholds. Other fields are ignored.
  void SetConfig(const OfflineSortformerDiarizationConfig &config);

  // Run diarization over a single-channel waveform.
  //
  // @param audio Pointer to a 1-D array of PCM samples in [-1, 1].
  // @param n Number of samples.
  OfflineSpeakerDiarizationResult Process(const float *audio, int32_t n) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_SORTFORMER_DIARIZATION_H_
