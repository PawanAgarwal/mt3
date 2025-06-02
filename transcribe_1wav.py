"""Offline MT3 transcription example.

This script demonstrates how to transcribe a local ``1.wav`` file using the
MT3 model. It mirrors the high level steps performed in the
``music_transcription_with_transformers.ipynb`` notebook and applies several
postprocessing steps such as overlapping window inference and note
de-duplication.
"""

from pathlib import Path
from importlib import resources
import functools
import numpy as np
import pandas as pd
import tensorflow as tf
import librosa
import jax
import gin
import seqio
import t5
import t5x

from mt3 import models, network, spectrograms, preprocessors
from mt3 import vocabularies, metrics_utils, note_sequences


class InferenceModel:
    """Wrapper around the T5X MT3 model for offline inference."""

    def __init__(self, checkpoint_path: str, model_type: str = "mt3"):
        if model_type == "ismir2021":
            num_velocity_bins = 127
            self.encoding_spec = note_sequences.NoteEncodingSpec
            self.inputs_length = 512
        elif model_type == "mt3":
            num_velocity_bins = 1
            self.encoding_spec = note_sequences.NoteEncodingWithTiesSpec
            self.inputs_length = 256
        else:
            raise ValueError(f"unknown model_type: {model_type}")

        gin_dir = resources.files("mt3") / "gin"
        gin_files = [
            str(gin_dir / "model.gin"),
            str(gin_dir / f"{model_type}.gin"),
        ]

        self.batch_size = 8
        self.outputs_length = 1024
        self.sequence_length = {"inputs": self.inputs_length, "targets": self.outputs_length}
        self.partitioner = t5x.partitioning.PjitPartitioner(num_partitions=1)

        self.spectrogram_config = spectrograms.SpectrogramConfig()
        self.codec = vocabularies.build_codec(
            vocab_config=vocabularies.VocabularyConfig(num_velocity_bins=num_velocity_bins)
        )
        self.vocabulary = vocabularies.vocabulary_from_codec(self.codec)
        self.output_features = {
            "inputs": seqio.ContinuousFeature(dtype=tf.float32, rank=2),
            "targets": seqio.Feature(vocabulary=self.vocabulary),
        }

        self._parse_gin(gin_files)
        self.model = self._load_model()
        self.restore_from_checkpoint(checkpoint_path)

    @property
    def input_shapes(self):
        return {
            "encoder_input_tokens": (self.batch_size, self.inputs_length),
            "decoder_input_tokens": (self.batch_size, self.outputs_length),
        }

    def _parse_gin(self, gin_files):
        gin_bindings = [
            "from __gin__ import dynamic_registration",
            "from mt3 import vocabularies",
            "VOCAB_CONFIG=@vocabularies.VocabularyConfig()",
            "vocabularies.VocabularyConfig.num_velocity_bins=%NUM_VELOCITY_BINS",
        ]
        with gin.unlock_config():
            gin.parse_config_files_and_bindings(gin_files, gin_bindings, finalize_config=False)

    def _load_model(self):
        model_cfg = gin.get_configurable(network.T5Config)()
        module = network.Transformer(config=model_cfg)
        return models.ContinuousInputsEncoderDecoderModel(
            module=module,
            input_vocabulary=self.output_features["inputs"].vocabulary,
            output_vocabulary=self.output_features["targets"].vocabulary,
            optimizer_def=t5x.adafactor.Adafactor(decay_rate=0.8, step_offset=0),
            input_depth=spectrograms.input_depth(self.spectrogram_config),
        )

    def restore_from_checkpoint(self, path: str) -> None:
        init = t5x.utils.TrainStateInitializer(
            optimizer_def=self.model.optimizer_def,
            init_fn=self.model.get_initial_variables,
            input_shapes=self.input_shapes,
            partitioner=self.partitioner,
        )
        restore_cfg = t5x.utils.RestoreCheckpointConfig(path=path, mode="specific", dtype="float32")
        train_state_axes = init.train_state_axes
        self._predict_fn = self._get_predict_fn(train_state_axes)
        self._train_state = init.from_checkpoint_or_scratch([restore_cfg], init_rng=jax.random.PRNGKey(0))

    @functools.lru_cache()
    def _get_predict_fn(self, train_state_axes):
        def partial_predict_fn(params, batch, decode_rng):
            return self.model.predict_batch_with_aux(params, batch, decoder_params={"decode_rng": None})

        return self.partitioner.partition(
            partial_predict_fn,
            in_axis_resources=(train_state_axes.params, t5x.partitioning.PartitionSpec("data"), None),
            out_axis_resources=t5x.partitioning.PartitionSpec("data"),
        )

    def predict_tokens(self, batch, seed=0):
        prediction, _ = self._predict_fn(self._train_state.params, batch, jax.random.PRNGKey(seed))
        return self.vocabulary.decode_tf(prediction).numpy()

    # ------------------------------------------------------------------
    # Audio / dataset utilities
    # ------------------------------------------------------------------
    def _audio_to_frames(self, audio: np.ndarray):
        frame_size = self.spectrogram_config.hop_width
        padding = [0, frame_size - len(audio) % frame_size]
        audio = np.pad(audio, padding, mode="constant")
        frames = spectrograms.split_audio(audio, self.spectrogram_config)
        num_frames = len(audio) // frame_size
        times = np.arange(num_frames) / self.spectrogram_config.frames_per_second
        return frames, times

    def audio_to_dataset(self, audio: np.ndarray) -> tf.data.Dataset:
        frames, frame_times = self._audio_to_frames(audio)
        return tf.data.Dataset.from_tensors({"inputs": frames, "input_times": frame_times})

    def preprocess(self, ds: tf.data.Dataset) -> tf.data.Dataset:
        pp_chain = [
            functools.partial(
                t5.data.preprocessors.split_tokens_to_inputs_length,
                sequence_length=self.sequence_length,
                output_features=self.output_features,
                feature_key="inputs",
                additional_feature_keys=["input_times"],
            ),
            preprocessors.add_dummy_targets,
            functools.partial(preprocessors.compute_spectrograms, spectrogram_config=self.spectrogram_config),
        ]
        for pp in pp_chain:
            ds = pp(ds)
        return ds

    def postprocess(self, tokens, example):
        tokens = self._trim_eos(tokens)
        start_time = example["input_times"][0]
        start_time -= start_time % (1 / self.codec.steps_per_second)
        return {"est_tokens": tokens, "start_time": start_time, "raw_inputs": []}

    @staticmethod
    def _trim_eos(tokens):
        tokens = np.array(tokens, np.int32)
        if vocabularies.DECODED_EOS_ID in tokens:
            tokens = tokens[: np.argmax(tokens == vocabularies.DECODED_EOS_ID)]
        return tokens

    # ------------------------------------------------------------------
    # Main inference entrypoint
    # ------------------------------------------------------------------
    def __call__(self, audio: np.ndarray):
        ds = self.audio_to_dataset(audio)
        ds = self.preprocess(ds)
        model_ds = self.model.FEATURE_CONVERTER_CLS(pack=False)(ds, task_feature_lengths=self.sequence_length)
        model_ds = model_ds.batch(self.batch_size)
        inferences = (
            tokens
            for batch in model_ds.as_numpy_iterator()
            for tokens in self.predict_tokens(batch)
        )
        predictions = []
        for example, tokens in zip(ds.as_numpy_iterator(), inferences):
            predictions.append(self.postprocess(tokens, example))
        result = metrics_utils.event_predictions_to_ns(predictions, codec=self.codec, encoding_spec=self.encoding_spec)
        return result["est_ns"]


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------

def load_audio(path: str, sr: int = 16000) -> np.ndarray:
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio


def segment_audio(audio: np.ndarray, config: spectrograms.SpectrogramConfig, window_frames: int = 256, hop_frames: int = 128):
    """Segment audio into overlapping windows of frames."""
    frames = spectrograms.split_audio(audio, config)
    segments = []
    times = []
    for start in range(0, len(frames) - window_frames + 1, hop_frames):
        seg = frames[start : start + window_frames]
        t = start / config.frames_per_second
        segments.append(seg)
        times.append(t)
    return np.array(segments), np.array(times)


def deduplicate(notes, diff_ms=50.0):
    notes = sorted(notes, key=lambda n: (n.pitch, n.start_time))
    deduped = []
    for n in notes:
        if deduped and n.pitch == deduped[-1].pitch and n.start_time - deduped[-1].start_time <= diff_ms / 1000.0:
            continue
        deduped.append(n)
    return deduped


def remove_short_notes(notes, min_ms=30.0):
    return [n for n in notes if (n.end_time - n.start_time) >= min_ms / 1000.0]


# ----------------------------------------------------------------------
# Main script
# ----------------------------------------------------------------------

def main():
    audio_path = Path(__file__).with_name("1.wav")
    audio = load_audio(str(audio_path))

    checkpoint_dir = Path("checkpoints/mt3_small_multiinst")
    model = InferenceModel(str(checkpoint_dir), "mt3")

    segments, start_times = segment_audio(audio, model.spectrogram_config)

    all_notes = []
    for segment, start in zip(segments, start_times):
        ns = model(segment)
        for note in ns.notes:
            note.start_time += start
            note.end_time += start
            all_notes.append(note)

    all_notes = deduplicate(all_notes)
    all_notes = remove_short_notes(all_notes)

    final_ns = note_sequences.NoteSequence()
    final_ns.notes.extend(all_notes)
    final_ns.total_time = max(n.end_time for n in all_notes) if all_notes else 0

    df = pd.DataFrame({
        "onset": [n.start_time for n in all_notes],
        "offset": [n.end_time for n in all_notes],
        "pitch": [n.pitch for n in all_notes],
        "velocity": [n.velocity for n in all_notes],
    })
    print(df.head())


if __name__ == "__main__":
    main()
