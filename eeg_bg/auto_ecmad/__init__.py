"""Extensible two-stage, EEGNet-guided ECMAD automation."""
from .pipeline import AutoECMADPipeline
from .types import DatasetAdapter, ModelTrainer, PairDecomposer, PreparedRecording, RecordingSource, TrialBatch

__all__ = ["AutoECMADPipeline", "DatasetAdapter", "ModelTrainer", "PairDecomposer",
           "PreparedRecording", "RecordingSource", "TrialBatch"]
