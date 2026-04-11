// sherpa-onnx/csrc/offline-sortformer-diarization.cc
//
// Copyright (c)  2026  Xiaomi Corporation
//
// C++ inference for NVIDIA NeMo Sortformer v2 streaming speaker diarization.
// Audio preprocessing, streaming state management and post-processing follow
// the reference implementation in parakeet-rs
// (https://github.com/OurWorldMedia/parakeet-rs), which itself mirrors the
// upstream NeMo SortformerEncLabelModel inference pipeline.

#include "sherpa-onnx/csrc/offline-sortformer-diarization.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "kaldi-native-fbank/csrc/rfft.h"
#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"

namespace sherpa_onnx {

// Mel / STFT constants — these mirror NeMo's AudioToMelSpectrogramPreprocessor
// configuration used by the public Sortformer checkpoints.
static constexpr int32_t kSortformerSampleRate = 16000;
static constexpr int32_t kSortformerNFft = 512;
static constexpr int32_t kSortformerWinLength = 400;  // 25 ms
static constexpr int32_t kSortformerHopLength = 160;  // 10 ms
static constexpr int32_t kSortformerNMels = 128;
static constexpr int32_t kSortformerNumSpeakers = 4;
static constexpr int32_t kSortformerEmbDim = 512;
static constexpr int32_t kSortformerSubsampling = 8;  // mel frames -> model frames
static constexpr float kSortformerPreemph = 0.97f;
static constexpr float kSortformerLogGuard = 5.960464478e-8f;  // 2^-24

// Streaming constants used as defaults when the ONNX metadata does not provide
// them. The public Sortformer v2 export bakes these in, but we read overrides
// from metadata just in case.
static constexpr int32_t kSortformerChunkLen = 124;
static constexpr int32_t kSortformerFifoLen = 124;
static constexpr int32_t kSortformerSpkcacheLen = 188;
static constexpr int32_t kSortformerSpkcacheUpdatePeriod = 188;
static constexpr int32_t kSortformerRightContext = 1;

// Frame duration in seconds after 8x subsampling.
static constexpr float kSortformerFrameDuration =
    static_cast<float>(kSortformerSubsampling * kSortformerHopLength) /
    static_cast<float>(kSortformerSampleRate);  // 80 ms

// Frame duration at the 10 ms audio-frame rate used by NeMo's VAD-style
// post-processing (after subsampling-factor repeat_interleave).
static constexpr float kSortformerAudioFrameDuration =
    static_cast<float>(kSortformerHopLength) /
    static_cast<float>(kSortformerSampleRate);  // 10 ms

// _compress_spkcache hyperparameters (NeMo defaults, training-only knobs
// omitted).
static constexpr float kSortformerPredScoreThreshold = 0.25f;
static constexpr float kSortformerSilThreshold = 0.2f;
static constexpr float kSortformerStrongBoostRate = 0.75f;
static constexpr float kSortformerWeakBoostRate = 1.5f;
static constexpr float kSortformerMinPosScoresRate = 0.5f;
static constexpr int32_t kSortformerSilFramesPerSpk = 3;
static constexpr float kSortformerScoresBoostLatest = 0.05f;
static constexpr int32_t kSortformerMaxIndex = 99999;

namespace {

using Seg = std::pair<float, float>;

// Port of NeMo's merge_overlap_segment. Sorts by start time, then merges any
// neighbours whose ranges touch or overlap.
static void MergeOverlapping(std::vector<Seg> *segs) {
  if (segs->size() < 2) {
    return;
  }
  std::sort(segs->begin(), segs->end(),
            [](const Seg &a, const Seg &b) { return a.first < b.first; });
  std::vector<Seg> out;
  out.reserve(segs->size());
  out.push_back((*segs)[0]);
  for (size_t i = 1; i < segs->size(); ++i) {
    if (out.back().second >= (*segs)[i].first) {
      if ((*segs)[i].second > out.back().second) {
        out.back().second = (*segs)[i].second;
      }
    } else {
      out.push_back((*segs)[i]);
    }
  }
  segs->swap(out);
}

// Port of NeMo's filter_short_segments — drop segments shorter than threshold.
static void FilterShort(std::vector<Seg> *segs, float threshold) {
  if (threshold <= 0.0f) {
    return;
  }
  segs->erase(std::remove_if(segs->begin(), segs->end(),
                             [threshold](const Seg &s) {
                               return (s.second - s.first) < threshold;
                             }),
              segs->end());
}

// Port of NeMo's get_gap_segments — returns inter-segment gap ranges.
// Assumes the input is already sorted by start time.
static std::vector<Seg> GapSegments(const std::vector<Seg> &segs) {
  std::vector<Seg> gaps;
  if (segs.size() < 2) {
    return gaps;
  }
  gaps.reserve(segs.size() - 1);
  for (size_t i = 1; i < segs.size(); ++i) {
    gaps.emplace_back(segs[i - 1].second, segs[i].first);
  }
  return gaps;
}

// Port of NeMo's binarization(): hysteresis thresholding with onset/offset
// plus pad_onset/pad_offset and a merge-overlap step.
static std::vector<Seg> BinarizePerSpeaker(const float *frames, int32_t n,
                                           float onset, float offset,
                                           float pad_onset, float pad_offset,
                                           float frame_length_in_sec) {
  std::vector<Seg> out;
  bool speech = false;
  float start = 0.0f;
  int32_t last_i = 0;
  for (int32_t i = 0; i < n; ++i) {
    float p = frames[i];
    last_i = i;
    if (speech) {
      if (p < offset) {
        float end_s = i * frame_length_in_sec + pad_offset;
        float start_s = std::max(0.0f, start - pad_onset);
        if (end_s > start_s) {
          out.emplace_back(start_s, end_s);
        }
        start = i * frame_length_in_sec;
        speech = false;
      }
    } else {
      if (p > onset) {
        start = i * frame_length_in_sec;
        speech = true;
      }
    }
  }
  if (speech) {
    float end_s = last_i * frame_length_in_sec + pad_offset;
    float start_s = std::max(0.0f, start - pad_onset);
    if (end_s > start_s) {
      out.emplace_back(start_s, end_s);
    }
  }
  MergeOverlapping(&out);
  return out;
}

// Port of NeMo's filtering() with filter_speech_first = 1.0.
static void FilterPerSpeaker(std::vector<Seg> *segs, float min_on,
                             float min_off) {
  if (segs->empty()) {
    return;
  }
  if (min_on > 0.0f) {
    FilterShort(segs, min_on);
  }
  if (min_off > 0.0f && segs->size() >= 2) {
    // Sort to match NeMo's behaviour on gap extraction.
    std::sort(segs->begin(), segs->end(),
              [](const Seg &a, const Seg &b) { return a.first < b.first; });
    std::vector<Seg> gaps = GapSegments(*segs);
    // NeMo keeps the *short* gaps (those smaller than min_off) and folds
    // them back in as speech, which effectively merges neighbouring segments
    // separated by a short gap.
    std::vector<Seg> short_gaps;
    short_gaps.reserve(gaps.size());
    for (const auto &g : gaps) {
      if ((g.second - g.first) < min_off) {
        short_gaps.push_back(g);
      }
    }
    segs->insert(segs->end(), short_gaps.begin(), short_gaps.end());
    MergeOverlapping(segs);
  }
}

}  // namespace

void OfflineSortformerDiarizationConfig::Register(ParseOptions *po) {
  model.Register(po);

  po->Register("sortformer-onset", &onset,
               "Sigmoid threshold to START a speaker segment.");

  po->Register("sortformer-offset", &offset,
               "Sigmoid threshold to END a speaker segment.");

  po->Register("sortformer-pad-onset", &pad_onset,
               "Extra duration (in seconds) added before each detected "
               "speech segment during post-processing.");

  po->Register("sortformer-pad-offset", &pad_offset,
               "Extra duration (in seconds) added after each detected "
               "speech segment during post-processing.");

  po->Register("sortformer-min-duration-on", &min_duration_on,
               "Segments shorter than this (in seconds) are discarded.");

  po->Register(
      "sortformer-min-duration-off", &min_duration_off,
      "If the gap between two segments of the same speaker is smaller "
      "than this (in seconds), the segments are merged.");
}

bool OfflineSortformerDiarizationConfig::Validate() const {
  if (!model.Validate()) {
    return false;
  }
  if (onset < 0.0f || onset > 1.0f) {
    SHERPA_ONNX_LOGE("sortformer-onset should be in [0, 1]. Given %f", onset);
    return false;
  }
  if (offset < 0.0f || offset > 1.0f) {
    SHERPA_ONNX_LOGE("sortformer-offset should be in [0, 1]. Given %f", offset);
    return false;
  }
  if (pad_onset < 0.0f) {
    SHERPA_ONNX_LOGE("sortformer-pad-onset must be non-negative. Given %f",
                     pad_onset);
    return false;
  }
  if (pad_offset < 0.0f) {
    SHERPA_ONNX_LOGE("sortformer-pad-offset must be non-negative. Given %f",
                     pad_offset);
    return false;
  }
  if (min_duration_on < 0.0f) {
    SHERPA_ONNX_LOGE(
        "sortformer-min-duration-on must be non-negative. Given %f",
        min_duration_on);
    return false;
  }
  if (min_duration_off < 0.0f) {
    SHERPA_ONNX_LOGE(
        "sortformer-min-duration-off must be non-negative. Given %f",
        min_duration_off);
    return false;
  }
  return true;
}

std::string OfflineSortformerDiarizationConfig::ToString() const {
  std::ostringstream os;

  os << "OfflineSortformerDiarizationConfig(";
  os << "model=" << model.ToString() << ", ";
  os << "onset=" << onset << ", ";
  os << "offset=" << offset << ", ";
  os << "pad_onset=" << pad_onset << ", ";
  os << "pad_offset=" << pad_offset << ", ";
  os << "min_duration_on=" << min_duration_on << ", ";
  os << "min_duration_off=" << min_duration_off << ")";

  return os.str();
}

// ----------------------------------------------------------------------------
// Implementation
// ----------------------------------------------------------------------------

class OfflineSortformerDiarization::Impl {
 public:
  explicit Impl(const OfflineSortformerDiarizationConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR),
        sess_opts_(GetSessionOptionsImpl(config.model.num_threads,
                                         config.model.provider)),
        allocator_{},
        rfft_(kSortformerNFft) {
    auto buf = ReadFile(config_.model.model);
    Init(buf.data(), buf.size());
    BuildHannWindow();
    BuildMelFilterbank();
  }

  int32_t SampleRate() const { return kSortformerSampleRate; }

  int32_t NumSpeakers() const { return kSortformerNumSpeakers; }

  void SetConfig(const OfflineSortformerDiarizationConfig &config) {
    config_.onset = config.onset;
    config_.offset = config.offset;
    config_.pad_onset = config.pad_onset;
    config_.pad_offset = config.pad_offset;
    config_.min_duration_on = config.min_duration_on;
    config_.min_duration_off = config.min_duration_off;
  }

  OfflineSpeakerDiarizationResult Process(const float *audio, int32_t n) {
    OfflineSpeakerDiarizationResult ans;
    if (n <= 0 || audio == nullptr) {
      return ans;
    }

    // 1. Extract log-mel features for the whole audio.
    std::vector<float> mel = ExtractMelFeatures(audio, n);
    // mel has shape (num_mel_frames, n_mels) row-major.
    int32_t num_mel_frames = static_cast<int32_t>(mel.size()) / kSortformerNMels;
    if (num_mel_frames <= 0) {
      return ans;
    }

    // 2. Reset streaming state.
    ResetState();

    // 3. Iterate the mel tensor in chunks. Each chunk feeds
    //    (chunk_len + right_context) * subsampling mel frames into the model,
    //    and we stride by chunk_len * subsampling mel frames.
    const int32_t feed_size =
        (chunk_len_ + right_context_) * kSortformerSubsampling;
    const int32_t stride = chunk_len_ * kSortformerSubsampling;

    std::vector<float> all_chunk_preds;  // flattened (num_frames, 4)
    all_chunk_preds.reserve(num_mel_frames / kSortformerSubsampling *
                            kSortformerNumSpeakers);

    int32_t num_chunks = (num_mel_frames + stride - 1) / stride;
    if (num_chunks == 0) {
      num_chunks = 1;
    }

    std::vector<float> chunk_feat(feed_size * kSortformerNMels, 0.0f);

    for (int32_t ci = 0; ci < num_chunks; ++ci) {
      int32_t start = ci * stride;
      int32_t end = std::min(start + feed_size, num_mel_frames);
      int32_t current_len = end - start;

      // Fill chunk_feat from mel[start..end], pad remainder with zeros.
      std::fill(chunk_feat.begin(), chunk_feat.end(), 0.0f);
      if (current_len > 0) {
        std::copy(mel.begin() + start * kSortformerNMels,
                  mel.begin() + end * kSortformerNMels, chunk_feat.begin());
      }

      std::vector<float> chunk_preds =
          StreamingUpdate(chunk_feat.data(), current_len);
      all_chunk_preds.insert(all_chunk_preds.end(), chunk_preds.begin(),
                             chunk_preds.end());
    }

    // 4. Clip the final prediction sequence to the audio length.
    int32_t total_model_frames =
        static_cast<int32_t>(all_chunk_preds.size()) / kSortformerNumSpeakers;
    int32_t max_model_frames =
        num_mel_frames / kSortformerSubsampling;  // floor
    if (max_model_frames < total_model_frames) {
      total_model_frames = max_model_frames;
      all_chunk_preds.resize(total_model_frames * kSortformerNumSpeakers);
    }
    if (total_model_frames <= 0) {
      return ans;
    }

    // 5. Binarize predictions into speaker segments.
    ans = Binarize(all_chunk_preds.data(), total_model_frames, n);
    return ans;
  }

 private:
  void Init(void *model_data, size_t model_data_length) {
    sess_ = std::make_unique<Ort::Session>(env_, model_data, model_data_length,
                                           sess_opts_);

    GetInputNames(sess_.get(), &input_names_, &input_names_ptr_);
    GetOutputNames(sess_.get(), &output_names_, &output_names_ptr_);

    Ort::ModelMetadata meta_data = sess_->GetModelMetadata();
    if (config_.model.debug) {
      std::ostringstream os;
      PrintModelMetadata(os, meta_data);
      SHERPA_ONNX_LOGE("%s", os.str().c_str());
    }

    // Read optional streaming overrides from the ONNX metadata. The public
    // Sortformer v2 export ships with these defaults, but parakeet-rs honors
    // metadata overrides so we do the same.
    chunk_len_ = ReadIntMetaData(meta_data, "chunk_len", kSortformerChunkLen);
    fifo_len_ = ReadIntMetaData(meta_data, "fifo_len", kSortformerFifoLen);
    spkcache_len_ =
        ReadIntMetaData(meta_data, "spkcache_len", kSortformerSpkcacheLen);
    right_context_ =
        ReadIntMetaData(meta_data, "right_context", kSortformerRightContext);
    spkcache_update_period_ = ReadIntMetaData(
        meta_data, "spkcache_update_period", kSortformerSpkcacheUpdatePeriod);
  }

  static int32_t ReadIntMetaData(Ort::ModelMetadata &meta,  // NOLINT
                                 const char *key, int32_t fallback) {
    Ort::AllocatorWithDefaultOptions allocator;
    auto value = meta.LookupCustomMetadataMapAllocated(key, allocator);
    if (!value) {
      return fallback;
    }
    try {
      return std::stoi(value.get());
    } catch (...) {
      return fallback;
    }
  }

  void BuildHannWindow() {
    // librosa-style Hann window of length win_length, zero-padded (centered)
    // to n_fft. This matches parakeet-rs and NeMo's FilterbankFeatures.
    window_.assign(kSortformerNFft, 0.0f);
    int32_t offset = (kSortformerNFft - kSortformerWinLength) / 2;
    for (int32_t i = 0; i < kSortformerWinLength; ++i) {
      // Periodic Hann (fftbins=True): divide by N, not N-1.
      float w = 0.5f - 0.5f * std::cos(2.0f * static_cast<float>(M_PI) *
                                       static_cast<float>(i) /
                                       static_cast<float>(kSortformerWinLength));
      window_[offset + i] = w;
    }
  }

  static double HzToMelSlaney(double hz) {
    constexpr double kFSp = 200.0 / 3.0;
    constexpr double kMinLogHz = 1000.0;
    constexpr double kMinLogMel = kMinLogHz / kFSp;
    constexpr double kLogStep = 0.06875177742094912;  // log(6.4) / 27.0
    if (hz < kMinLogHz) {
      return hz / kFSp;
    }
    return kMinLogMel + std::log(hz / kMinLogHz) / kLogStep;
  }

  static double MelToHzSlaney(double mel) {
    constexpr double kFSp = 200.0 / 3.0;
    constexpr double kMinLogHz = 1000.0;
    constexpr double kMinLogMel = kMinLogHz / kFSp;
    constexpr double kLogStep = 0.06875177742094912;
    if (mel < kMinLogMel) {
      return mel * kFSp;
    }
    return kMinLogHz * std::exp((mel - kMinLogMel) * kLogStep);
  }

  void BuildMelFilterbank() {
    // Slaney-style triangular mel filterbank with Slaney normalization, as
    // used by librosa (htk=False) and NeMo's default mel setup.
    int32_t freq_bins = kSortformerNFft / 2 + 1;
    mel_basis_.assign(kSortformerNMels * freq_bins, 0.0f);

    double fmax = kSortformerSampleRate / 2.0;
    double mel_min = HzToMelSlaney(0.0);
    double mel_max = HzToMelSlaney(fmax);

    std::vector<double> mel_points(kSortformerNMels + 2);
    for (int32_t i = 0; i < static_cast<int32_t>(mel_points.size()); ++i) {
      double mel =
          mel_min + (mel_max - mel_min) * static_cast<double>(i) /
                        static_cast<double>(kSortformerNMels + 1);
      mel_points[i] = MelToHzSlaney(mel);
    }

    std::vector<double> fft_freqs(freq_bins);
    for (int32_t k = 0; k < freq_bins; ++k) {
      fft_freqs[k] = static_cast<double>(k) *
                     static_cast<double>(kSortformerSampleRate) /
                     static_cast<double>(kSortformerNFft);
    }

    std::vector<double> fdiff(kSortformerNMels + 1);
    for (int32_t i = 0; i < kSortformerNMels + 1; ++i) {
      fdiff[i] = mel_points[i + 1] - mel_points[i];
    }

    for (int32_t i = 0; i < kSortformerNMels; ++i) {
      double enorm = 2.0 / (mel_points[i + 2] - mel_points[i]);
      for (int32_t k = 0; k < freq_bins; ++k) {
        double lower = (fft_freqs[k] - mel_points[i]) / fdiff[i];
        double upper = (mel_points[i + 2] - fft_freqs[k]) / fdiff[i + 1];
        double v = std::max(0.0, std::min(lower, upper)) * enorm;
        mel_basis_[i * freq_bins + k] = static_cast<float>(v);
      }
    }
  }

  // Applies preemphasis and returns a new buffer.
  static std::vector<float> ApplyPreemphasis(const float *audio, int32_t n) {
    std::vector<float> out(static_cast<size_t>(n));
    if (n <= 0) return out;
    out[0] = audio[0];
    for (int32_t i = 1; i < n; ++i) {
      out[i] = audio[i] - kSortformerPreemph * audio[i - 1];
    }
    return out;
  }

  // Compute a (num_mel_frames, n_mels) log-mel spectrogram row-major.
  std::vector<float> ExtractMelFeatures(const float *audio, int32_t n) {
    // 1. Preemphasis.
    std::vector<float> preemph = ApplyPreemphasis(audio, n);

    // 2. librosa-style center=True padding: pad n_fft/2 with zeros on both
    // sides.
    int32_t pad = kSortformerNFft / 2;
    std::vector<float> padded(static_cast<size_t>(n + 2 * pad), 0.0f);
    std::copy(preemph.begin(), preemph.end(), padded.begin() + pad);

    // 3. Frame the signal. num_frames = floor((len - n_fft) / hop) + 1.
    int32_t total = static_cast<int32_t>(padded.size());
    int32_t num_frames = 0;
    if (total >= kSortformerNFft) {
      num_frames = (total - kSortformerNFft) / kSortformerHopLength + 1;
    }
    if (num_frames <= 0) {
      return {};
    }

    int32_t freq_bins = kSortformerNFft / 2 + 1;
    std::vector<float> out(static_cast<size_t>(num_frames) * kSortformerNMels,
                           0.0f);

    std::vector<float> fft_buf(kSortformerNFft);
    std::vector<float> power(freq_bins);

    for (int32_t t = 0; t < num_frames; ++t) {
      int32_t start = t * kSortformerHopLength;
      // Multiply input frame by the zero-padded Hann window.
      for (int32_t i = 0; i < kSortformerNFft; ++i) {
        fft_buf[i] = padded[start + i] * window_[i];
      }
      // In-place real FFT. After Compute, fft_buf[0]=R[0], fft_buf[1]=R[n/2],
      // fft_buf[2k]=R[k], fft_buf[2k+1]=I[k] for 1 <= k < n/2.
      rfft_.Compute(fft_buf.data());

      power[0] = fft_buf[0] * fft_buf[0];
      power[freq_bins - 1] = fft_buf[1] * fft_buf[1];
      for (int32_t k = 1; k < freq_bins - 1; ++k) {
        float r = fft_buf[2 * k];
        float im = fft_buf[2 * k + 1];
        power[k] = r * r + im * im;
      }

      // Apply mel filterbank then log with guard.
      float *out_row = out.data() + t * kSortformerNMels;
      for (int32_t m = 0; m < kSortformerNMels; ++m) {
        const float *row = mel_basis_.data() + m * freq_bins;
        float acc = 0.0f;
        for (int32_t k = 0; k < freq_bins; ++k) {
          acc += row[k] * power[k];
        }
        out_row[m] = std::log(acc + kSortformerLogGuard);
      }
    }

    return out;
  }

  void ResetState() {
    spkcache_.clear();
    spkcache_frames_ = 0;
    spkcache_preds_.clear();
    spkcache_preds_initialized_ = false;

    fifo_.clear();
    fifo_frames_ = 0;
    fifo_preds_.clear();

    mean_sil_emb_.assign(kSortformerEmbDim, 0.0f);
    n_sil_frames_ = 0;
  }

  // Run one streaming_update step.
  //
  // chunk_feat points to a (feed_size, n_mels) row-major tensor. current_len
  // is the number of valid mel frames (the rest are zero-padded).
  //
  // Returns the chunk predictions for the chunk_len model frames only (or
  // fewer if this is the last, partial chunk).
  std::vector<float> StreamingUpdate(const float *chunk_feat,
                                     int32_t current_len) {
    const int32_t feed_size =
        (chunk_len_ + right_context_) * kSortformerSubsampling;
    const int32_t total_prefix = spkcache_frames_ + fifo_frames_;

    // Build input tensors. For variable-length dims we use the current cache
    // and FIFO sizes.
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::array<int64_t, 3> chunk_shape = {1, feed_size, kSortformerNMels};
    Ort::Value chunk_value = Ort::Value::CreateTensor<float>(
        memory_info, const_cast<float *>(chunk_feat), feed_size * kSortformerNMels,
        chunk_shape.data(), chunk_shape.size());

    std::array<int64_t, 1> len_shape = {1};
    int64_t chunk_len_val = current_len;
    Ort::Value chunk_lengths_value = Ort::Value::CreateTensor<int64_t>(
        memory_info, &chunk_len_val, 1, len_shape.data(), len_shape.size());

    // spkcache: (1, spkcache_frames_, emb_dim)
    std::array<int64_t, 3> spkcache_shape = {1, spkcache_frames_,
                                             kSortformerEmbDim};
    if (spkcache_.empty()) {
      // Ort::Value::CreateTensor rejects nullptr; give it a dummy 1-element
      // buffer with shape {1, 0, emb_dim}. ONNX Runtime is fine with zero-sized
      // dims if the pointer isn't null.
      spkcache_.assign(1, 0.0f);
    }
    Ort::Value spkcache_value = Ort::Value::CreateTensor<float>(
        memory_info, spkcache_.data(), spkcache_frames_ * kSortformerEmbDim,
        spkcache_shape.data(), spkcache_shape.size());

    int64_t spkcache_len_val = spkcache_frames_;
    Ort::Value spkcache_lengths_value = Ort::Value::CreateTensor<int64_t>(
        memory_info, &spkcache_len_val, 1, len_shape.data(), len_shape.size());

    std::array<int64_t, 3> fifo_shape = {1, fifo_frames_, kSortformerEmbDim};
    if (fifo_.empty()) {
      fifo_.assign(1, 0.0f);
    }
    Ort::Value fifo_value = Ort::Value::CreateTensor<float>(
        memory_info, fifo_.data(), fifo_frames_ * kSortformerEmbDim,
        fifo_shape.data(), fifo_shape.size());

    int64_t fifo_len_val = fifo_frames_;
    Ort::Value fifo_lengths_value = Ort::Value::CreateTensor<int64_t>(
        memory_info, &fifo_len_val, 1, len_shape.data(), len_shape.size());

    // The Sortformer ONNX export uses 6 named inputs in this order:
    //   chunk, chunk_lengths, spkcache, spkcache_lengths, fifo, fifo_lengths
    // We look the indices up from the actual input_names_ rather than
    // hard-coding the order, since different exports may reorder them.
    std::array<Ort::Value, 6> input_values{
        std::move(chunk_value),         std::move(chunk_lengths_value),
        std::move(spkcache_value),      std::move(spkcache_lengths_value),
        std::move(fifo_value),          std::move(fifo_lengths_value)};
    static const std::array<const char *, 6> kKnownInputNames = {
        "chunk", "chunk_lengths", "spkcache", "spkcache_lengths", "fifo",
        "fifo_lengths"};

    std::vector<Ort::Value> inputs_reordered;
    inputs_reordered.reserve(input_names_.size());
    for (const auto &name : input_names_) {
      int32_t idx = -1;
      for (int32_t i = 0; i < static_cast<int32_t>(kKnownInputNames.size());
           ++i) {
        if (name == kKnownInputNames[i]) {
          idx = i;
          break;
        }
      }
      if (idx < 0) {
        SHERPA_ONNX_LOGE("Sortformer ONNX has an unexpected input name: %s",
                         name.c_str());
        SHERPA_ONNX_EXIT(-1);
      }
      inputs_reordered.push_back(std::move(input_values[idx]));
    }

    auto outputs = sess_->Run(
        {}, input_names_ptr_.data(), inputs_reordered.data(),
        inputs_reordered.size(), output_names_ptr_.data(), output_names_.size());

    // Resolve output indices by name. We need:
    //  - spkcache_fifo_chunk_preds: (1, S+F+T', 4)
    //  - chunk_pre_encode_embs:     (1, T', 512)
    //  - chunk_pre_encode_lengths:  (1,)    (not used here)
    int32_t preds_idx = -1;
    int32_t embs_idx = -1;
    for (int32_t i = 0; i < static_cast<int32_t>(output_names_.size()); ++i) {
      if (output_names_[i] == "spkcache_fifo_chunk_preds") {
        preds_idx = i;
      } else if (output_names_[i] == "chunk_pre_encode_embs") {
        embs_idx = i;
      }
    }
    if (preds_idx < 0 || embs_idx < 0) {
      SHERPA_ONNX_LOGE(
          "Sortformer ONNX export is missing expected outputs. Needed "
          "spkcache_fifo_chunk_preds and chunk_pre_encode_embs.");
      SHERPA_ONNX_EXIT(-1);
    }

    auto preds_info = outputs[preds_idx].GetTensorTypeAndShapeInfo();
    auto preds_shape_vec = preds_info.GetShape();
    const float *preds_data = outputs[preds_idx].GetTensorData<float>();
    int32_t preds_batch = static_cast<int32_t>(preds_shape_vec[0]);
    int32_t preds_rows = static_cast<int32_t>(preds_shape_vec[1]);
    int32_t preds_cols = static_cast<int32_t>(preds_shape_vec[2]);
    (void)preds_batch;
    if (preds_cols != kSortformerNumSpeakers) {
      SHERPA_ONNX_LOGE(
          "Expected Sortformer to emit %d speakers per frame, got %d",
          kSortformerNumSpeakers, preds_cols);
      SHERPA_ONNX_EXIT(-1);
    }

    auto embs_info = outputs[embs_idx].GetTensorTypeAndShapeInfo();
    auto embs_shape_vec = embs_info.GetShape();
    const float *embs_data = outputs[embs_idx].GetTensorData<float>();
    int32_t embs_rows = static_cast<int32_t>(embs_shape_vec[1]);
    int32_t embs_cols = static_cast<int32_t>(embs_shape_vec[2]);
    if (embs_cols != kSortformerEmbDim) {
      SHERPA_ONNX_LOGE(
          "Expected Sortformer to emit %d-dim embeddings, got %d",
          kSortformerEmbDim, embs_cols);
      SHERPA_ONNX_EXIT(-1);
    }

    // The prediction tensor concatenates spkcache + fifo + chunk along the
    // time axis. We only care about the chunk part, and within that we drop
    // the right_context lookahead frames.
    int32_t chunk_model_frames = preds_rows - total_prefix;
    int32_t valid_model_frames =
        (current_len + kSortformerSubsampling - 1) / kSortformerSubsampling;
    int32_t keep = std::min({chunk_len_, chunk_model_frames, valid_model_frames,
                             embs_rows});
    if (keep < 0) {
      keep = 0;
    }

    std::vector<float> chunk_preds(static_cast<size_t>(keep) *
                                       kSortformerNumSpeakers,
                                   0.0f);
    for (int32_t t = 0; t < keep; ++t) {
      int32_t src_row = total_prefix + t;
      for (int32_t s = 0; s < kSortformerNumSpeakers; ++s) {
        chunk_preds[t * kSortformerNumSpeakers + s] =
            preds_data[src_row * preds_cols + s];
      }
    }

    // Update streaming state: append chunk embs to FIFO and refresh fifo_preds
    // from the current model output, then spill oldest rows into the speaker
    // cache when the FIFO is full. Follows NeMo's streaming_update step by
    // step.
    const int32_t old_spkcache_frames =
        static_cast<int32_t>(spkcache_frames_);
    const int32_t old_fifo_frames = static_cast<int32_t>(fifo_frames_);
    const int32_t chunk_frames = keep;
    const int32_t n_spk = kSortformerNumSpeakers;

    // Append chunk embeddings to the FIFO (embeddings are never refreshed).
    int32_t new_fifo_frames = old_fifo_frames + chunk_frames;
    std::vector<float> new_fifo(static_cast<size_t>(new_fifo_frames) *
                                kSortformerEmbDim);
    if (old_fifo_frames > 0) {
      std::copy(fifo_.begin(),
                fifo_.begin() + old_fifo_frames * kSortformerEmbDim,
                new_fifo.begin());
    }
    for (int32_t t = 0; t < chunk_frames; ++t) {
      std::copy(embs_data + t * embs_cols, embs_data + (t + 1) * embs_cols,
                new_fifo.begin() + (old_fifo_frames + t) * kSortformerEmbDim);
    }
    fifo_.swap(new_fifo);

    // Rebuild fifo_preds from the current iteration's output (NeMo overwrites
    // the old fifo_preds with the fresh model prediction before concatenating
    // the new chunk preds).
    std::vector<float> new_fifo_preds(static_cast<size_t>(new_fifo_frames) *
                                      n_spk);
    for (int32_t t = 0; t < new_fifo_frames; ++t) {
      int32_t src_row = old_spkcache_frames + t;
      for (int32_t s = 0; s < n_spk; ++s) {
        new_fifo_preds[t * n_spk + s] =
            preds_data[src_row * preds_cols + s];
      }
    }
    fifo_preds_.swap(new_fifo_preds);
    fifo_frames_ = new_fifo_frames;

    if (old_fifo_frames + chunk_frames > fifo_len_) {
      int32_t pop_out_len = spkcache_update_period_;
      pop_out_len =
          std::max(pop_out_len, chunk_frames - fifo_len_ + old_fifo_frames);
      pop_out_len =
          std::min(pop_out_len, old_fifo_frames + chunk_frames);
      pop_out_len =
          std::min(pop_out_len, static_cast<int32_t>(fifo_frames_));
      if (pop_out_len < 0) pop_out_len = 0;

      if (pop_out_len > 0) {
        // Save the popped rows before mutating the FIFO.
        std::vector<float> pop_embs(
            fifo_.begin(),
            fifo_.begin() + pop_out_len * kSortformerEmbDim);
        std::vector<float> pop_preds(
            fifo_preds_.begin(), fifo_preds_.begin() + pop_out_len * n_spk);

        // Update running silence-embedding profile from the popped rows.
        UpdateSilenceProfile(pop_embs.data(), pop_preds.data(), pop_out_len);

        // Drop the popped rows from the FIFO.
        int32_t rem = static_cast<int32_t>(fifo_frames_) - pop_out_len;
        std::vector<float> tfifo(static_cast<size_t>(rem) * kSortformerEmbDim);
        std::copy(fifo_.begin() + pop_out_len * kSortformerEmbDim,
                  fifo_.begin() + (pop_out_len + rem) * kSortformerEmbDim,
                  tfifo.begin());
        fifo_.swap(tfifo);
        std::vector<float> tfifop(static_cast<size_t>(rem) * n_spk);
        std::copy(fifo_preds_.begin() + pop_out_len * n_spk,
                  fifo_preds_.begin() + (pop_out_len + rem) * n_spk,
                  tfifop.begin());
        fifo_preds_.swap(tfifop);
        fifo_frames_ = rem;

        // Append the popped embeddings to the speaker cache.
        int32_t new_cache_frames = old_spkcache_frames + pop_out_len;
        std::vector<float> new_cache(
            static_cast<size_t>(new_cache_frames) * kSortformerEmbDim);
        if (old_spkcache_frames > 0) {
          std::copy(
              spkcache_.begin(),
              spkcache_.begin() + old_spkcache_frames * kSortformerEmbDim,
              new_cache.begin());
        }
        std::copy(pop_embs.begin(), pop_embs.end(),
                  new_cache.begin() + old_spkcache_frames * kSortformerEmbDim);
        spkcache_.swap(new_cache);

        // Only append popped preds to spkcache_preds if it was already seeded.
        if (spkcache_preds_initialized_) {
          std::vector<float> new_cache_preds(
              static_cast<size_t>(new_cache_frames) * n_spk);
          if (old_spkcache_frames > 0) {
            std::copy(
                spkcache_preds_.begin(),
                spkcache_preds_.begin() + old_spkcache_frames * n_spk,
                new_cache_preds.begin());
          }
          std::copy(pop_preds.begin(), pop_preds.end(),
                    new_cache_preds.begin() + old_spkcache_frames * n_spk);
          spkcache_preds_.swap(new_cache_preds);
        }
        spkcache_frames_ = new_cache_frames;

        if (spkcache_frames_ > spkcache_len_) {
          if (!spkcache_preds_initialized_) {
            // Seed spkcache_preds_ on the very first compression: the first
            // old_spkcache_frames rows come from the current iteration's
            // prediction for the spkcache slice, followed by the popped rows.
            spkcache_preds_.assign(
                static_cast<size_t>(new_cache_frames) * n_spk, 0.0f);
            for (int32_t t = 0; t < old_spkcache_frames; ++t) {
              for (int32_t s = 0; s < n_spk; ++s) {
                spkcache_preds_[t * n_spk + s] =
                    preds_data[t * preds_cols + s];
              }
            }
            std::copy(pop_preds.begin(), pop_preds.end(),
                      spkcache_preds_.begin() + old_spkcache_frames * n_spk);
            spkcache_preds_initialized_ = true;
          }
          CompressSpkcache();
        }
      }
    }

    return chunk_preds;
  }

  // Convert a (num_frames, 4) sigmoid prediction tensor into speaker segments
  // using NeMo's ts_vad_post_processing pipeline: repeat each model frame by
  // the subsampling factor, then run per-speaker binarization + filtering.
  OfflineSpeakerDiarizationResult Binarize(const float *preds,
                                           int32_t num_frames,
                                           int32_t num_audio_samples) const {
    OfflineSpeakerDiarizationResult ans;

    float onset = config_.onset;
    float offset = config_.offset;
    if (offset > onset) {
      offset = onset;
    }
    float pad_onset = config_.pad_onset;
    float pad_offset = config_.pad_offset;
    float min_on = config_.min_duration_on;
    float min_off = config_.min_duration_off;

    float audio_duration = static_cast<float>(num_audio_samples) /
                           static_cast<float>(kSortformerSampleRate);

    int32_t upsample = kSortformerSubsampling;
    int32_t n_audio_frames = num_frames * upsample;
    std::vector<float> channel(static_cast<size_t>(n_audio_frames), 0.0f);

    for (int32_t spk = 0; spk < kSortformerNumSpeakers; ++spk) {
      // repeat_interleave: each model frame becomes `upsample` audio frames.
      for (int32_t t = 0; t < num_frames; ++t) {
        float p = preds[t * kSortformerNumSpeakers + spk];
        float *dst = channel.data() + t * upsample;
        for (int32_t k = 0; k < upsample; ++k) {
          dst[k] = p;
        }
      }

      std::vector<Seg> segs =
          BinarizePerSpeaker(channel.data(), n_audio_frames, onset, offset,
                             pad_onset, pad_offset,
                             kSortformerAudioFrameDuration);
      FilterPerSpeaker(&segs, min_on, min_off);

      for (auto &s : segs) {
        if (s.second > audio_duration) {
          s.second = audio_duration;
        }
        if (s.first < 0.0f) {
          s.first = 0.0f;
        }
        if (s.second > s.first) {
          ans.Add(OfflineSpeakerDiarizationSegment(s.first, s.second, spk));
        }
      }
    }

    return ans;
  }

  // Update running mean of silence embeddings. NeMo considers a frame
  // "silent" when the sum of its per-speaker sigmoid predictions is below
  // kSortformerSilThreshold.
  void UpdateSilenceProfile(const float *embs, const float *preds, int32_t n) {
    if (n <= 0) return;
    if (mean_sil_emb_.size() !=
        static_cast<size_t>(kSortformerEmbDim)) {
      mean_sil_emb_.assign(kSortformerEmbDim, 0.0f);
    }
    int64_t new_sil = 0;
    std::vector<float> sum(kSortformerEmbDim, 0.0f);
    for (int32_t t = 0; t < n; ++t) {
      float total = 0.0f;
      for (int32_t s = 0; s < kSortformerNumSpeakers; ++s) {
        total += preds[t * kSortformerNumSpeakers + s];
      }
      if (total < kSortformerSilThreshold) {
        new_sil++;
        const float *row = embs + t * kSortformerEmbDim;
        for (int32_t i = 0; i < kSortformerEmbDim; ++i) {
          sum[i] += row[i];
        }
      }
    }
    if (new_sil == 0) return;
    int64_t total_frames = n_sil_frames_ + new_sil;
    float scale = 1.0f / static_cast<float>(total_frames);
    for (int32_t i = 0; i < kSortformerEmbDim; ++i) {
      float old_sum =
          mean_sil_emb_[i] * static_cast<float>(n_sil_frames_);
      mean_sil_emb_[i] = (old_sum + sum[i]) * scale;
    }
    n_sil_frames_ = total_frames;
  }

  // Port of NeMo's _get_log_pred_scores. Produces an (n, n_spk) row-major
  // score tensor that is high for confident single-speaker frames.
  static void GetLogPredScores(const float *preds, int32_t n, int32_t n_spk,
                               float *scores) {
    const float thresh = kSortformerPredScoreThreshold;
    const float log2 = std::log(2.0f);
    for (int32_t t = 0; t < n; ++t) {
      float b[kSortformerNumSpeakers];
      float sum_b = 0.0f;
      for (int32_t s = 0; s < n_spk; ++s) {
        float p = preds[t * n_spk + s];
        float b_s = std::log(std::max(1.0f - p, thresh));
        b[s] = b_s;
        sum_b += b_s;
      }
      for (int32_t s = 0; s < n_spk; ++s) {
        float p = preds[t * n_spk + s];
        float a = std::log(std::max(p, thresh));
        scores[t * n_spk + s] = a - b[s] + sum_b + log2;
      }
    }
  }

  // Port of NeMo's _disable_low_scores.
  static void DisableLowScores(const float *preds, int32_t n, int32_t n_spk,
                               int32_t min_pos_scores_per_spk, float *scores) {
    const float neg_inf = -std::numeric_limits<float>::infinity();
    for (int32_t t = 0; t < n; ++t) {
      for (int32_t s = 0; s < n_spk; ++s) {
        if (preds[t * n_spk + s] <= 0.5f) {
          scores[t * n_spk + s] = neg_inf;
        }
      }
    }
    std::vector<int32_t> pos_count(n_spk, 0);
    for (int32_t t = 0; t < n; ++t) {
      for (int32_t s = 0; s < n_spk; ++s) {
        if (scores[t * n_spk + s] > 0.0f) ++pos_count[s];
      }
    }
    for (int32_t s = 0; s < n_spk; ++s) {
      if (pos_count[s] < min_pos_scores_per_spk) continue;
      for (int32_t t = 0; t < n; ++t) {
        float v = scores[t * n_spk + s];
        if (v <= 0.0f && preds[t * n_spk + s] > 0.5f) {
          scores[t * n_spk + s] = neg_inf;
        }
      }
    }
  }

  // Port of NeMo's _boost_topk_scores. For each speaker independently, find
  // the top n_boost frames and bump their scores by scale_factor * log(2).
  static void BoostTopkScores(float *scores, int32_t n, int32_t n_spk,
                              int32_t n_boost, float scale_factor) {
    if (n_boost <= 0 || n <= 0) return;
    const float delta = scale_factor * std::log(2.0f);
    std::vector<std::pair<float, int32_t>> pairs;
    pairs.reserve(n);
    for (int32_t s = 0; s < n_spk; ++s) {
      pairs.clear();
      for (int32_t t = 0; t < n; ++t) {
        pairs.emplace_back(scores[t * n_spk + s], t);
      }
      int32_t k = std::min(n_boost, n);
      if (k < n) {
        std::nth_element(
            pairs.begin(), pairs.begin() + k, pairs.end(),
            [](const std::pair<float, int32_t> &a,
               const std::pair<float, int32_t> &b) { return a.first > b.first; });
      }
      for (int32_t i = 0; i < k; ++i) {
        int32_t t = pairs[i].second;
        scores[t * n_spk + s] += delta;  // -inf + delta stays -inf
      }
    }
  }

  // Port of NeMo's _get_topk_indices. Returns spkcache_len_ ordered time
  // indices plus a mask marking slots that should be replaced with the mean
  // silence embedding.
  void GetTopkIndices(const float *scores, int32_t n_frames_data, int32_t n_spk,
                      std::vector<int32_t> *time_indices,
                      std::vector<uint8_t> *is_disabled) const {
    const int32_t n_sil = kSortformerSilFramesPerSpk;
    const int32_t n_frames_total = n_frames_data + n_sil;
    const float pos_inf = std::numeric_limits<float>::infinity();
    const float neg_inf = -pos_inf;

    std::vector<std::pair<float, int32_t>> pairs;
    pairs.reserve(static_cast<size_t>(n_frames_total) * n_spk);
    for (int32_t s = 0; s < n_spk; ++s) {
      int32_t base = s * n_frames_total;
      for (int32_t t = 0; t < n_frames_data; ++t) {
        pairs.emplace_back(scores[t * n_spk + s], base + t);
      }
      for (int32_t k = 0; k < n_sil; ++k) {
        pairs.emplace_back(pos_inf, base + n_frames_data + k);
      }
    }

    int32_t total = static_cast<int32_t>(pairs.size());
    int32_t take = std::min(spkcache_len_, total);
    if (take < total) {
      std::nth_element(pairs.begin(), pairs.begin() + take, pairs.end(),
                       [](const std::pair<float, int32_t> &a,
                          const std::pair<float, int32_t> &b) {
                         return a.first > b.first;
                       });
    }
    pairs.resize(take);

    // Push any -inf entries to the end of the sorted output so they cluster
    // at the tail, matching NeMo's "replace with max_index before sort" trick.
    for (int32_t i = 0; i < take; ++i) {
      if (pairs[i].first == neg_inf) {
        pairs[i].second = kSortformerMaxIndex;
      }
    }
    std::sort(pairs.begin(), pairs.end(),
              [](const std::pair<float, int32_t> &a,
                 const std::pair<float, int32_t> &b) {
                return a.second < b.second;
              });

    time_indices->assign(spkcache_len_, 0);
    is_disabled->assign(spkcache_len_, 0);
    for (int32_t k = 0; k < spkcache_len_; ++k) {
      if (k >= take) {
        (*is_disabled)[k] = 1;
        continue;
      }
      int32_t fi = pairs[k].second;
      bool disabled = (fi == kSortformerMaxIndex);
      int32_t time_idx = 0;
      if (!disabled) {
        time_idx = fi % n_frames_total;
        if (time_idx >= n_frames_data) {
          disabled = true;
        }
      }
      (*time_indices)[k] = disabled ? 0 : time_idx;
      (*is_disabled)[k] = disabled ? 1 : 0;
    }
  }

  // Port of NeMo's _gather_spkcache_and_preds. Disabled slots resolve to the
  // mean silence embedding + zero predictions.
  void GatherSpkcacheAndPreds(const std::vector<int32_t> &time_indices,
                              const std::vector<uint8_t> &is_disabled,
                              std::vector<float> *new_cache,
                              std::vector<float> *new_preds) const {
    const int32_t n_spk = kSortformerNumSpeakers;
    new_cache->assign(static_cast<size_t>(spkcache_len_) * kSortformerEmbDim,
                      0.0f);
    new_preds->assign(static_cast<size_t>(spkcache_len_) * n_spk, 0.0f);
    for (int32_t k = 0; k < spkcache_len_; ++k) {
      float *out_emb = new_cache->data() + k * kSortformerEmbDim;
      float *out_pred = new_preds->data() + k * n_spk;
      if (is_disabled[k]) {
        if (mean_sil_emb_.size() ==
            static_cast<size_t>(kSortformerEmbDim)) {
          std::copy(mean_sil_emb_.begin(), mean_sil_emb_.end(), out_emb);
        }
        // preds row already zero-initialised.
      } else {
        int32_t t = time_indices[k];
        std::copy(spkcache_.begin() + t * kSortformerEmbDim,
                  spkcache_.begin() + (t + 1) * kSortformerEmbDim, out_emb);
        std::copy(spkcache_preds_.begin() + t * n_spk,
                  spkcache_preds_.begin() + (t + 1) * n_spk, out_pred);
      }
    }
  }

  // Top-level port of NeMo's _compress_spkcache. Rewrites spkcache_ /
  // spkcache_preds_ in place down to spkcache_len_ rows using the same
  // quality-scored top-k selection as the reference implementation.
  void CompressSpkcache() {
    const int32_t n_spk = kSortformerNumSpeakers;
    const int32_t n_frames = static_cast<int32_t>(spkcache_frames_);
    const int32_t spkcache_len_per_spk =
        spkcache_len_ / n_spk - kSortformerSilFramesPerSpk;
    const int32_t strong_boost_per_spk = static_cast<int32_t>(std::floor(
        static_cast<float>(spkcache_len_per_spk) * kSortformerStrongBoostRate));
    const int32_t weak_boost_per_spk = static_cast<int32_t>(std::floor(
        static_cast<float>(spkcache_len_per_spk) * kSortformerWeakBoostRate));
    const int32_t min_pos_scores_per_spk = static_cast<int32_t>(std::floor(
        static_cast<float>(spkcache_len_per_spk) *
        kSortformerMinPosScoresRate));

    std::vector<float> scores(static_cast<size_t>(n_frames) * n_spk, 0.0f);
    GetLogPredScores(spkcache_preds_.data(), n_frames, n_spk, scores.data());
    DisableLowScores(spkcache_preds_.data(), n_frames, n_spk,
                     min_pos_scores_per_spk, scores.data());

    // Boost the most recently added frames slightly so that they aren't
    // immediately evicted by older, better-scored frames.
    if (kSortformerScoresBoostLatest > 0.0f && n_frames > spkcache_len_) {
      for (int32_t t = spkcache_len_; t < n_frames; ++t) {
        for (int32_t s = 0; s < n_spk; ++s) {
          scores[t * n_spk + s] += kSortformerScoresBoostLatest;
        }
      }
    }

    BoostTopkScores(scores.data(), n_frames, n_spk, strong_boost_per_spk,
                    2.0f);
    BoostTopkScores(scores.data(), n_frames, n_spk, weak_boost_per_spk, 1.0f);

    std::vector<int32_t> time_indices;
    std::vector<uint8_t> is_disabled;
    GetTopkIndices(scores.data(), n_frames, n_spk, &time_indices, &is_disabled);

    std::vector<float> new_cache;
    std::vector<float> new_preds;
    GatherSpkcacheAndPreds(time_indices, is_disabled, &new_cache, &new_preds);

    spkcache_.swap(new_cache);
    spkcache_preds_.swap(new_preds);
    spkcache_frames_ = spkcache_len_;
  }

 private:
  OfflineSortformerDiarizationConfig config_;
  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;

  std::unique_ptr<Ort::Session> sess_;
  std::vector<std::string> input_names_;
  std::vector<const char *> input_names_ptr_;
  std::vector<std::string> output_names_;
  std::vector<const char *> output_names_ptr_;

  knf::Rfft rfft_;
  std::vector<float> window_;     // (n_fft,)
  std::vector<float> mel_basis_;  // (n_mels, n_fft/2 + 1) row-major

  // Streaming state.
  std::vector<float> spkcache_;  // (spkcache_frames_, emb_dim) row-major
  int64_t spkcache_frames_ = 0;
  // (spkcache_frames_, num_spk) row-major. Tracks the predictions associated
  // with the embeddings in spkcache_, used by _compress_spkcache.
  std::vector<float> spkcache_preds_;
  bool spkcache_preds_initialized_ = false;

  std::vector<float> fifo_;  // (fifo_frames_, emb_dim) row-major
  int64_t fifo_frames_ = 0;
  // (fifo_frames_, num_spk) row-major. Tracks the predictions associated with
  // the embeddings in fifo_, used when spilling into the speaker cache.
  std::vector<float> fifo_preds_;

  // Running mean of "silence" embeddings, i.e. embeddings whose total speaker
  // activity is below kSortformerSilThreshold. Used to substitute for disabled
  // slots when compressing the speaker cache.
  std::vector<float> mean_sil_emb_;  // (emb_dim,)
  int64_t n_sil_frames_ = 0;

  // Streaming hyperparameters read from model metadata.
  int32_t chunk_len_ = kSortformerChunkLen;
  int32_t fifo_len_ = kSortformerFifoLen;
  int32_t spkcache_len_ = kSortformerSpkcacheLen;
  int32_t right_context_ = kSortformerRightContext;
  int32_t spkcache_update_period_ = kSortformerSpkcacheUpdatePeriod;
};

// ----------------------------------------------------------------------------
// Public API
// ----------------------------------------------------------------------------

OfflineSortformerDiarization::OfflineSortformerDiarization(
    const OfflineSortformerDiarizationConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

OfflineSortformerDiarization::~OfflineSortformerDiarization() = default;

int32_t OfflineSortformerDiarization::SampleRate() const {
  return impl_->SampleRate();
}

int32_t OfflineSortformerDiarization::NumSpeakers() const {
  return impl_->NumSpeakers();
}

void OfflineSortformerDiarization::SetConfig(
    const OfflineSortformerDiarizationConfig &config) {
  impl_->SetConfig(config);
}

OfflineSpeakerDiarizationResult OfflineSortformerDiarization::Process(
    const float *audio, int32_t n) const {
  return impl_->Process(audio, n);
}

}  // namespace sherpa_onnx
