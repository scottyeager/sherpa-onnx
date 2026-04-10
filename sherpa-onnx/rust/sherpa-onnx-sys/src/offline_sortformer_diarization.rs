#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]

use std::os::raw::{c_char, c_float};

use crate::offline_speaker_diarization::OfflineSpeakerDiarizationResult;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OfflineSortformerDiarizationModelConfig {
    pub model: *const c_char,
    pub num_threads: i32,
    pub debug: i32,
    pub provider: *const c_char,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OfflineSortformerDiarizationConfig {
    pub model: OfflineSortformerDiarizationModelConfig,
    pub onset: c_float,
    pub offset: c_float,
    pub min_duration_on: c_float,
    pub min_duration_off: c_float,
    pub pad_onset: c_float,
    pub pad_offset: c_float,
}

#[repr(C)]
pub struct OfflineSortformerDiarization {
    _private: [u8; 0],
}

extern "C" {
    pub fn SherpaOnnxCreateOfflineSortformerDiarization(
        config: *const OfflineSortformerDiarizationConfig,
    ) -> *const OfflineSortformerDiarization;

    pub fn SherpaOnnxDestroyOfflineSortformerDiarization(
        sd: *const OfflineSortformerDiarization,
    );

    pub fn SherpaOnnxOfflineSortformerDiarizationGetSampleRate(
        sd: *const OfflineSortformerDiarization,
    ) -> i32;

    pub fn SherpaOnnxOfflineSortformerDiarizationGetNumSpeakers(
        sd: *const OfflineSortformerDiarization,
    ) -> i32;

    pub fn SherpaOnnxOfflineSortformerDiarizationSetConfig(
        sd: *const OfflineSortformerDiarization,
        config: *const OfflineSortformerDiarizationConfig,
    );

    pub fn SherpaOnnxOfflineSortformerDiarizationProcess(
        sd: *const OfflineSortformerDiarization,
        samples: *const c_float,
        n: i32,
    ) -> *const OfflineSpeakerDiarizationResult;
}
