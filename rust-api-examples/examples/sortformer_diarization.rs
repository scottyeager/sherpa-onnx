//! NVIDIA Sortformer end-to-end speaker diarization example.
//!
//! Unlike the Pyannote diarization example, Sortformer is a single model that
//! directly emits per-frame speaker activity probabilities — no separate
//! speaker embedding extractor or clustering stage is required. The public
//! `diar_streaming_sortformer_4spk-v2` checkpoint supports up to 4 speakers.
//!
//! To run it, first grab an exported ONNX model (see
//! <https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2> for the
//! upstream checkpoint and ONNX export instructions), then:
//!
//! ```sh
//! cargo run --example sortformer_diarization -- \
//!     ./diar_streaming_sortformer_4spk-v2.onnx \
//!     ./some-audio.wav
//! ```

use sherpa_onnx::{
    OfflineSortformerDiarization, OfflineSortformerDiarizationConfig,
    OfflineSortformerDiarizationModelConfig, Wave,
};

fn main() {
    let mut args = std::env::args().skip(1);
    let model_path = args.next().unwrap_or_else(|| {
        "./diar_streaming_sortformer_4spk-v2.onnx".to_string()
    });
    let wav_path = args
        .next()
        .unwrap_or_else(|| "./0-four-speakers-zh.wav".to_string());

    // The default post-processing thresholds already match NVIDIA's CallHome
    // recipe, so no threshold overrides are needed here.
    let config = OfflineSortformerDiarizationConfig {
        model: OfflineSortformerDiarizationModelConfig {
            model: Some(model_path),
            num_threads: 2,
            ..Default::default()
        },
        ..Default::default()
    };

    let sd = OfflineSortformerDiarization::create(&config)
        .expect("Failed to initialize Sortformer diarization");

    let wave = Wave::read(&wav_path).expect("Failed to read wave");

    assert_eq!(
        sd.sample_rate(),
        wave.sample_rate(),
        "Unexpected sample rate (Sortformer expects {} Hz)",
        sd.sample_rate(),
    );

    println!("Model supports {} speakers", sd.num_speakers());

    let result = sd
        .process(wave.samples())
        .expect("Failed to run Sortformer diarization");

    println!("Number of speakers: {}", result.num_speakers());
    println!("Number of segments: {}", result.num_segments());

    for s in result.sort_by_start_time() {
        println!("{:.3} -- {:.3} speaker_{:02}", s.start, s.end, s.speaker);
    }
}
