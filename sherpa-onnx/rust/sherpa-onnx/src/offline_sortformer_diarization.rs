//! NVIDIA Sortformer end-to-end speaker diarization.
//!
//! Sortformer is a single-model diarization pipeline: given 16 kHz mono audio
//! it emits per-frame speaker activity probabilities for a fixed number of
//! speakers (4 in the public `diar_streaming_sortformer_4spk-v2` checkpoint).
//! Unlike the Pyannote pipeline wrapped by
//! `crate::OfflineSpeakerDiarization`, there is no separate speaker
//! embedding model or clustering stage — just point this at the exported
//! ONNX file and call [`OfflineSortformerDiarization::process`].
//!
//! See `rust-api-examples/examples/sortformer_diarization.rs` for a full
//! example.

use crate::{
    offline_speaker_diarization::OfflineSpeakerDiarizationSegment,
    utils::to_c_ptr,
};
use sherpa_onnx_sys as sys;
use std::ffi::CString;
use std::slice;

/// Sortformer model configuration (path + ONNX runtime options).
#[derive(Clone, Debug)]
pub struct OfflineSortformerDiarizationModelConfig {
    /// Path to the exported Sortformer ONNX file, e.g.
    /// `diar_streaming_sortformer_4spk-v2.onnx`.
    pub model: Option<String>,
    pub num_threads: i32,
    pub debug: bool,
    /// Execution provider such as `cpu`, `cuda`, or `coreml`.
    pub provider: Option<String>,
}

impl Default for OfflineSortformerDiarizationModelConfig {
    fn default() -> Self {
        Self {
            model: None,
            num_threads: 1,
            debug: false,
            provider: Some("cpu".to_string()),
        }
    }
}

impl OfflineSortformerDiarizationModelConfig {
    fn to_sys(
        &self,
        cstrings: &mut Vec<CString>,
    ) -> sys::OfflineSortformerDiarizationModelConfig {
        sys::OfflineSortformerDiarizationModelConfig {
            model: to_c_ptr(&self.model, cstrings),
            num_threads: self.num_threads,
            debug: self.debug as i32,
            provider: to_c_ptr(&self.provider, cstrings),
        }
    }
}

/// Top-level configuration for [`OfflineSortformerDiarization`].
///
/// Default values mirror NVIDIA's CallHome post-processing recipe
/// (`diar_streaming_sortformer_4spk-v2_callhome-part1.yaml`) so the
/// out-of-the-box output matches NeMo's reference pipeline on that data.
#[derive(Clone, Debug)]
pub struct OfflineSortformerDiarizationConfig {
    pub model: OfflineSortformerDiarizationModelConfig,
    /// Sigmoid threshold to start a speaker segment.
    pub onset: f32,
    /// Sigmoid threshold to end a speaker segment.
    pub offset: f32,
    /// Extra duration in seconds added before each detected speech segment.
    pub pad_onset: f32,
    /// Extra duration in seconds added after each detected speech segment.
    pub pad_offset: f32,
    /// Segments shorter than this duration (seconds) are dropped.
    pub min_duration_on: f32,
    /// Gaps smaller than this duration (seconds) between two segments of the
    /// same speaker cause the segments to be merged.
    pub min_duration_off: f32,
    /// Per-speaker median filter window (in 80 ms model frames) applied to
    /// the sigmoid predictions before binarization. Matches NeMo's default
    /// `median_window=11` for the callhome post-processing config. Set to
    /// `1` to disable.
    pub median_window: i32,
}

impl Default for OfflineSortformerDiarizationConfig {
    fn default() -> Self {
        Self {
            model: Default::default(),
            onset: 0.641,
            offset: 0.561,
            pad_onset: 0.229,
            pad_offset: 0.079,
            min_duration_on: 0.511,
            min_duration_off: 0.296,
            median_window: 11,
        }
    }
}

impl OfflineSortformerDiarizationConfig {
    fn to_sys(
        &self,
        cstrings: &mut Vec<CString>,
    ) -> sys::OfflineSortformerDiarizationConfig {
        sys::OfflineSortformerDiarizationConfig {
            model: self
                .model
                .to_sys(cstrings),
            onset: self.onset,
            offset: self.offset,
            min_duration_on: self.min_duration_on,
            min_duration_off: self.min_duration_off,
            pad_onset: self.pad_onset,
            pad_offset: self.pad_offset,
            median_window: self.median_window,
        }
    }
}

/// End-to-end NVIDIA Sortformer speaker diarizer.
pub struct OfflineSortformerDiarization {
    ptr: *const sys::OfflineSortformerDiarization,
}

unsafe impl Send for OfflineSortformerDiarization {}

impl OfflineSortformerDiarization {
    /// Load a Sortformer model and prepare it for inference.
    pub fn create(config: &OfflineSortformerDiarizationConfig) -> Option<Self> {
        let mut cstrings = Vec::new();
        let sys_config = config.to_sys(&mut cstrings);
        let ptr = unsafe { sys::SherpaOnnxCreateOfflineSortformerDiarization(&sys_config) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    /// Return the sample rate expected by the model (always 16000).
    pub fn sample_rate(&self) -> i32 {
        unsafe { sys::SherpaOnnxOfflineSortformerDiarizationGetSampleRate(self.ptr) }
    }

    /// Return the number of speakers the model supports (4 for the public
    /// Sortformer v2 export).
    pub fn num_speakers(&self) -> i32 {
        unsafe { sys::SherpaOnnxOfflineSortformerDiarizationGetNumSpeakers(self.ptr) }
    }

    /// Update post-processing thresholds without reloading the model. Only
    /// the `onset`, `offset`, `min_duration_on`, and `min_duration_off` fields
    /// are used — the model path is ignored.
    pub fn set_config(&self, config: &OfflineSortformerDiarizationConfig) {
        let mut cstrings = Vec::new();
        let sys_config = config.to_sys(&mut cstrings);
        unsafe { sys::SherpaOnnxOfflineSortformerDiarizationSetConfig(self.ptr, &sys_config) }
    }

    /// Run diarization over a complete single-channel waveform sampled at
    /// 16 kHz. Returns `None` if the underlying C call failed.
    pub fn process(&self, samples: &[f32]) -> Option<SortformerDiarizationResult> {
        let ptr = unsafe {
            sys::SherpaOnnxOfflineSortformerDiarizationProcess(
                self.ptr,
                samples.as_ptr(),
                samples.len() as i32,
            )
        };
        if ptr.is_null() {
            None
        } else {
            Some(SortformerDiarizationResult { ptr })
        }
    }
}

impl Drop for OfflineSortformerDiarization {
    fn drop(&mut self) {
        unsafe {
            if !self
                .ptr
                .is_null()
            {
                sys::SherpaOnnxDestroyOfflineSortformerDiarization(self.ptr);
            }
        }
    }
}

/// Diarization result returned from [`OfflineSortformerDiarization::process`].
///
/// Internally this wraps the same opaque C type used by the Pyannote pipeline,
/// so the accessor surface mirrors [`OfflineSpeakerDiarizationResult`].
pub struct SortformerDiarizationResult {
    ptr: *const sys::OfflineSpeakerDiarizationResult,
}

impl SortformerDiarizationResult {
    /// Number of distinct speakers detected in the recording.
    pub fn num_speakers(&self) -> i32 {
        unsafe { sys::SherpaOnnxOfflineSpeakerDiarizationResultGetNumSpeakers(self.ptr) }
    }

    /// Number of diarization segments.
    pub fn num_segments(&self) -> i32 {
        unsafe { sys::SherpaOnnxOfflineSpeakerDiarizationResultGetNumSegments(self.ptr) }
    }

    /// Segments sorted by start time.
    pub fn sort_by_start_time(&self) -> Vec<OfflineSpeakerDiarizationSegment> {
        let n = self.num_segments();
        if n <= 0 {
            return Vec::new();
        }

        unsafe {
            let p = sys::SherpaOnnxOfflineSpeakerDiarizationResultSortByStartTime(self.ptr);
            if p.is_null() {
                return Vec::new();
            }

            let segments = slice::from_raw_parts(p, n as usize)
                .iter()
                .map(|s| OfflineSpeakerDiarizationSegment {
                    start: s.start,
                    end: s.end,
                    speaker: s.speaker,
                })
                .collect::<Vec<_>>();
            sys::SherpaOnnxOfflineSpeakerDiarizationDestroySegment(p);
            segments
        }
    }
}

impl Drop for SortformerDiarizationResult {
    fn drop(&mut self) {
        unsafe {
            if !self
                .ptr
                .is_null()
            {
                sys::SherpaOnnxOfflineSpeakerDiarizationDestroyResult(self.ptr);
            }
        }
    }
}

