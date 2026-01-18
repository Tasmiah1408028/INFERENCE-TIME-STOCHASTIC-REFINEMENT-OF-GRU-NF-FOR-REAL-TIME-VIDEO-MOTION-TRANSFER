from typing import List, Optional

import torch

from gluonts.core.component import validated
from gluonts.dataset.field_names import FieldName
from gluonts.time_feature import TimeFeature
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.torch.util import copy_parameters
from gluonts.model.predictor import Predictor
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.transform import (
    Transformation,
    Chain,
    InstanceSplitter,
    ValidationSplitSampler,
    TestSplitSampler,
    ExpectedNumInstanceSampler,
    RenameFields,
    AsNumpyArray,
    ExpandDimArray,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    VstackFeatures,
    SetFieldIfNotPresent,
    TargetDimIndicator,
)

#from pts import Trainer
from pts.feature import (
    fourier_time_features_from_frequency,
    lags_for_fourier_time_features_from_frequency,
)
from pts.model.utils import get_module_forward_input_names

import sys
# Add the directory containing your custom TempFlowEstimator to the Python path

from estimator import PyTorchEstimator
from trainer import Trainer

from tempflow_network_SNF import TempFlowTrainingNetwork, TempFlowPredictionNetwork

#only gru
# class TempFlowEstimator(PyTorchEstimator):
#     @validated()
#     def __init__(
#         self,
#         input_size: int,
#         freq: str,
#         prediction_length: int,
#         target_dim: int,
#         trainer: Trainer = Trainer(),
#         context_length: Optional[int] = None,
#         num_layers: int = 3,
#         num_cells: int = 256,
#         cell_type: str = "GRU",
#         num_parallel_samples: int = 100,
#         dropout_rate: float = 0.1,
#         scaling: bool = True,
#         pick_incomplete: bool = False,
#         **kwargs,
#     ) -> None:
#         super().__init__(trainer=trainer, **kwargs)

#         self.freq = freq
#         self.context_length = context_length if context_length is not None else prediction_length
#         self.input_size = input_size
#         self.prediction_length = prediction_length
#         self.target_dim = target_dim
#         self.num_layers = num_layers
#         self.num_cells = num_cells
#         self.cell_type = cell_type
#         #self.num_parallel_samples = num_parallel_samples
#         self.dropout_rate = dropout_rate
#         self.scaling = scaling
#         self.pick_incomplete = pick_incomplete

#         # Define history length based on context_length
#         #self.history_length = self.context_length

#         self.train_sampler = ExpectedNumInstanceSampler(
#             num_instances=1.0,
#             min_past=0 if pick_incomplete else self.context_length,
#             min_future=prediction_length,
#         )

#         self.validation_sampler = ValidationSplitSampler(
#             min_past=0 if pick_incomplete else self.context_length,
#             min_future=prediction_length,
#         )

#     def create_transformation(self) -> Transformation:
#         return Chain(
#             [
#                 AsNumpyArray(
#                     field=FieldName.TARGET,
#                     expected_ndim=2,
#                 ),
#                 ExpandDimArray(
#                     field=FieldName.TARGET,
#                     axis=None,
#                 ),
#                 AddObservedValuesIndicator(
#                     target_field=FieldName.TARGET,
#                     output_field=FieldName.OBSERVED_VALUES,
#                 ),
#                 SetFieldIfNotPresent(field=FieldName.FEAT_STATIC_CAT, value=[0]),
#                 TargetDimIndicator(
#                     field_name="target_dimension_indicator",
#                     target_field=FieldName.TARGET,
#                 ),
#                 AsNumpyArray(field=FieldName.FEAT_STATIC_CAT, expected_ndim=1),
#             ]
#         )

#     def create_instance_splitter(self, mode: str):
#         assert mode in ["training", "validation", "test"], "Invalid mode for instance splitter."

#         instance_sampler = {
#             "training": self.train_sampler,
#             "validation": self.validation_sampler,
#             "test": TestSplitSampler(),
#         }[mode]

#         return InstanceSplitter(
#             target_field=FieldName.TARGET,
#             is_pad_field=FieldName.IS_PAD,
#             start_field=FieldName.START,
#             forecast_start_field=FieldName.FORECAST_START,
#             instance_sampler=instance_sampler,
#             past_length=self.context_length,
#             future_length=self.prediction_length,
#             time_series_fields=[
#                 FieldName.OBSERVED_VALUES,
#             ],
#         )

#     def create_training_network(self, device: torch.device) -> TempFlowTrainingNetwork:
#         return TempFlowTrainingNetwork(
#             input_size=self.input_size,
#             target_dim=self.target_dim,
#             num_layers=self.num_layers,
#             num_cells=self.num_cells,
#             cell_type=self.cell_type,
#             #history_length=self.history_length,
#             context_length=self.context_length,
#             prediction_length=self.prediction_length,
#             dropout_rate=self.dropout_rate,
#             scaling=self.scaling,
#         ).to(device)

#     def create_predictor(
#         self,
#         transformation: Transformation,
#         trained_network: TempFlowTrainingNetwork,
#         device: torch.device,
#     ) -> Predictor:
#         prediction_network = TempFlowPredictionNetwork(
#             input_size=self.input_size,
#             target_dim=self.target_dim,
#             num_layers=self.num_layers,
#             num_cells=self.num_cells,
#             cell_type=self.cell_type,
#             #history_length=self.history_length,
#             context_length=self.context_length,
#             prediction_length=self.prediction_length,
#             dropout_rate=self.dropout_rate,
#             scaling=self.scaling,
#         ).to(device)

#         copy_parameters(trained_network, prediction_network)
#         input_names = get_module_forward_input_names(prediction_network)
#         prediction_splitter = self.create_instance_splitter("test")

#         return PyTorchPredictor(
#             input_transform=transformation + prediction_splitter,
#             input_names=input_names,
#             prediction_net=prediction_network,
#             batch_size=self.trainer.batch_size,
#             freq=self.freq,
#             prediction_length=self.prediction_length,
#             device=device,
#         )

class TempFlowEstimator(PyTorchEstimator):
    @validated()
    def __init__(
        self,
        input_size: int,
        freq: str,
        prediction_length: int,
        target_dim: int,
        trainer: Trainer = Trainer(),
        context_length: Optional[int] = None,
        num_layers: int = 3,
        num_cells: int = 256,
        cell_type: str = "LSTM",
        num_parallel_samples: int = 100,
        dropout_rate: float = 0.1,
        cardinality: List[int] = [],
        embedding_dimension: int = 0,  # Adjust based on embedding removal
        flow_type="RealNVP",
        n_blocks=4,
        hidden_size=64,
        n_hidden=3,
        conditioning_length: int = 12,
        dequantize: bool = False,
        scaling: bool = True,
        pick_incomplete: bool = False,
        # Removed lags_seq and time_features
        **kwargs,
    ) -> None:
        super().__init__(trainer=trainer, **kwargs)

        self.freq = freq
        self.context_length = context_length if context_length is not None else prediction_length
        self.input_size = input_size
        self.prediction_length = prediction_length
        self.target_dim = target_dim
        self.num_layers = num_layers
        self.num_cells = num_cells
        self.cell_type = cell_type
        self.num_parallel_samples = num_parallel_samples
        self.dropout_rate = dropout_rate
        self.cardinality = cardinality
        self.embedding_dimension = embedding_dimension
        self.flow_type = flow_type
        self.n_blocks = n_blocks
        self.hidden_size = hidden_size
        self.n_hidden = n_hidden
        self.conditioning_length = conditioning_length
        self.dequantize = dequantize
        self.scaling = scaling

        # Adjusted history_length to remove lags_seq dependency
        self.history_length = self.context_length
        self.pick_incomplete = pick_incomplete

        self.train_sampler = ExpectedNumInstanceSampler(
            num_instances=1.0,
            min_past=0 if pick_incomplete else self.history_length,
            min_future=prediction_length,
        )

        self.validation_sampler = ValidationSplitSampler(
            min_past=0 if pick_incomplete else self.history_length,
            min_future=prediction_length,
        )

    def create_transformation(self) -> Transformation:
        return Chain(
            [
                AsNumpyArray(
                    field=FieldName.TARGET,
                    expected_ndim=2,
                ),
                ExpandDimArray(
                    field=FieldName.TARGET,
                    axis=None,
                ),
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                ),
                # Removed AddTimeFeatures as you're not using time features anymore
                SetFieldIfNotPresent(field=FieldName.FEAT_STATIC_CAT, value=[0]),
                TargetDimIndicator(
                    field_name="target_dimension_indicator",
                    target_field=FieldName.TARGET,
                ),
                AsNumpyArray(field=FieldName.FEAT_STATIC_CAT, expected_ndim=1),
            ]
        )

    def create_instance_splitter(self, mode: str):
        assert mode in ["training", "validation", "test"]

        instance_sampler = {
            "training": self.train_sampler,
            "validation": self.validation_sampler,
            "test": TestSplitSampler(),
        }[mode]

        return InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=instance_sampler,
            past_length=self.history_length,
            future_length=self.prediction_length,
            time_series_fields=[
                FieldName.OBSERVED_VALUES,
            ],
        ) + (
            RenameFields(
                {
                    f"past_{FieldName.TARGET}": f"past_{FieldName.TARGET}_cdf",
                    f"future_{FieldName.TARGET}": f"future_{FieldName.TARGET}_cdf",
                }
            )
        )

    def create_training_network(self, device: torch.device) -> TempFlowTrainingNetwork:
        return TempFlowTrainingNetwork(
            input_size=self.input_size,
            target_dim=self.target_dim,
            num_layers=self.num_layers,
            num_cells=self.num_cells,
            cell_type=self.cell_type,
            history_length=self.history_length,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            dropout_rate=self.dropout_rate,
            cardinality=self.cardinality,
            embedding_dimension=self.embedding_dimension,
            lags_seq=[],  # Provide an empty list to satisfy validation
            scaling=self.scaling,
            flow_type=self.flow_type,
            n_blocks=self.n_blocks,
            hidden_size=self.hidden_size,
            n_hidden=self.n_hidden,
            conditioning_length=self.conditioning_length,
            dequantize=self.dequantize,
        ).to(device)


    def create_predictor(
        self,
        transformation: Transformation,
        trained_network: TempFlowTrainingNetwork,
        device: torch.device,
    ) -> Predictor:
        prediction_network = TempFlowPredictionNetwork(
            input_size=self.input_size,
            target_dim=self.target_dim,
            num_layers=self.num_layers,
            num_cells=self.num_cells,
            cell_type=self.cell_type,
            history_length=self.history_length,
            context_length=self.context_length,
            prediction_length=self.prediction_length,
            dropout_rate=self.dropout_rate,
            cardinality=self.cardinality,
            embedding_dimension=self.embedding_dimension,
            lags_seq=[],  # Explicitly pass an empty lags_seq here too
            scaling=self.scaling,
            flow_type=self.flow_type,
            n_blocks=self.n_blocks,
            hidden_size=self.hidden_size,
            n_hidden=self.n_hidden,
            conditioning_length=self.conditioning_length,
            dequantize=self.dequantize,
            num_parallel_samples=self.num_parallel_samples,
        ).to(device)

        copy_parameters(trained_network, prediction_network)
        input_names = get_module_forward_input_names(prediction_network)
        prediction_splitter = self.create_instance_splitter("test")
    
        return PyTorchPredictor(
            input_transform=transformation + prediction_splitter,
            input_names=input_names,
            prediction_net=prediction_network,
            batch_size=self.trainer.batch_size,
            freq=self.freq,
            prediction_length=self.prediction_length,
            device=device,
        )

#with time features
# class TempFlowEstimator(PyTorchEstimator):
#     @validated()
#     def __init__(
#         self,
#         input_size: int,
#         freq: str,
#         prediction_length: int,
#         target_dim: int,
#         trainer: Trainer = Trainer(),
#         context_length: Optional[int] = None,
#         num_layers: int = 3,
#         num_cells: int = 256,
#         cell_type: str = "LSTM",
#         num_parallel_samples: int = 100,
#         dropout_rate: float = 0.1,
#         flow_type="RealNVP",
#         n_blocks=3,
#         hidden_size=100,
#         n_hidden=3,
#         conditioning_length: int = 200,
#         dequantize: bool = False,
#         scaling: bool = True,
#         pick_incomplete: bool = False,
#         lags_seq: Optional[List[int]] = None,
#         time_features: Optional[List[TimeFeature]] = None,
#         **kwargs,
#     ) -> None:
#         super().__init__(trainer=trainer, **kwargs)

#         self.freq = freq
#         self.context_length = (
#             context_length if context_length is not None else prediction_length
#         )

#         self.input_size = input_size
#         self.prediction_length = prediction_length
#         self.target_dim = target_dim
#         self.num_layers = num_layers
#         self.num_cells = num_cells
#         self.cell_type = cell_type
#         self.num_parallel_samples = num_parallel_samples
#         self.dropout_rate = dropout_rate

#         self.flow_type = flow_type
#         self.n_blocks = n_blocks
#         self.hidden_size = hidden_size
#         self.n_hidden = n_hidden
#         self.conditioning_length = conditioning_length
#         self.dequantize = dequantize

#         self.lags_seq = (
#             lags_seq
#             if lags_seq is not None
#             else lags_for_fourier_time_features_from_frequency(freq_str=freq)
#         )

#         self.time_features = (
#             time_features
#             if time_features is not None
#             else fourier_time_features_from_frequency(self.freq)
#         )

#         self.history_length = self.context_length + max(self.lags_seq)
#         self.pick_incomplete = pick_incomplete
#         self.scaling = scaling

#         self.train_sampler = ExpectedNumInstanceSampler(
#             num_instances=1.0,
#             min_past=0 if pick_incomplete else self.history_length,
#             min_future=prediction_length,
#         )

#         self.validation_sampler = ValidationSplitSampler(
#             min_past=0 if pick_incomplete else self.history_length,
#             min_future=prediction_length,
#         )

#     def create_transformation(self) -> Transformation:
#         return Chain(
#             [
#                 AsNumpyArray(
#                     field=FieldName.TARGET,
#                     expected_ndim=2,
#                 ),
#                 # maps the target to (1, T)
#                 # if the target data is uni dimensional
#                 ExpandDimArray(
#                     field=FieldName.TARGET,
#                     axis=None,
#                 ),
#                 AddObservedValuesIndicator(
#                     target_field=FieldName.TARGET,
#                     output_field=FieldName.OBSERVED_VALUES,
#                 ),
#                 AddTimeFeatures(
#                     start_field=FieldName.START,
#                     target_field=FieldName.TARGET,
#                     output_field=FieldName.FEAT_TIME,
#                     time_features=self.time_features,
#                     pred_length=self.prediction_length,
#                 ),
#                 VstackFeatures(
#                     output_field=FieldName.FEAT_TIME,
#                     input_fields=[FieldName.FEAT_TIME],
#                 ),
#                 AsNumpyArray(field=FieldName.FEAT_STATIC_CAT, expected_ndim=1),
#             ]
#         )

#     def create_instance_splitter(self, mode: str):
#         assert mode in ["training", "validation", "test"]

#         instance_sampler = {
#             "training": self.train_sampler,
#             "validation": self.validation_sampler,
#             "test": TestSplitSampler(),
#         }[mode]

#         return InstanceSplitter(
#             target_field=FieldName.TARGET,
#             is_pad_field=FieldName.IS_PAD,
#             start_field=FieldName.START,
#             forecast_start_field=FieldName.FORECAST_START,
#             instance_sampler=instance_sampler,
#             past_length=self.history_length,
#             future_length=self.prediction_length,
#             time_series_fields=[
#                 FieldName.FEAT_TIME,
#                 FieldName.OBSERVED_VALUES,
#             ],
#         )

#     def create_training_network(self, device: torch.device) -> TempFlowTrainingNetwork:
#         return TempFlowTrainingNetwork(
#             input_size=self.input_size,
#             target_dim=self.target_dim,
#             num_layers=self.num_layers,
#             num_cells=self.num_cells,
#             cell_type=self.cell_type,
#             history_length=self.history_length,
#             context_length=self.context_length,
#             prediction_length=self.prediction_length,
#             dropout_rate=self.dropout_rate,
#             lags_seq=self.lags_seq,
#             scaling=self.scaling,
#             flow_type=self.flow_type,
#             n_blocks=self.n_blocks,
#             hidden_size=self.hidden_size,
#             n_hidden=self.n_hidden,
#             conditioning_length=self.conditioning_length,
#             dequantize=self.dequantize,
#         ).to(device)

#     def create_predictor(
#         self,
#         transformation: Transformation,
#         trained_network: TempFlowTrainingNetwork,
#         device: torch.device,
#     ) -> Predictor:
#         prediction_network = TempFlowPredictionNetwork(
#             input_size=self.input_size,
#             target_dim=self.target_dim,
#             num_layers=self.num_layers,
#             num_cells=self.num_cells,
#             cell_type=self.cell_type,
#             history_length=self.history_length,
#             context_length=self.context_length,
#             prediction_length=self.prediction_length,
#             dropout_rate=self.dropout_rate,
#             lags_seq=self.lags_seq,
#             scaling=self.scaling,
#             flow_type=self.flow_type,
#             n_blocks=self.n_blocks,
#             hidden_size=self.hidden_size,
#             n_hidden=self.n_hidden,
#             conditioning_length=self.conditioning_length,
#             dequantize=self.dequantize,
#             num_parallel_samples=self.num_parallel_samples,
#         ).to(device)

#         copy_parameters(trained_network, prediction_network)
#         input_names = get_module_forward_input_names(prediction_network)
#         prediction_splitter = self.create_instance_splitter("test")

#         return PyTorchPredictor(
#             input_transform=transformation + prediction_splitter,
#             input_names=input_names,
#             prediction_net=prediction_network,
#             batch_size=self.trainer.batch_size,
#             freq=self.freq,
#             prediction_length=self.prediction_length,
#             device=device,
#         )

#with lags seq
# class TempFlowEstimator(PyTorchEstimator):
#     @validated()
#     def __init__(
#         self,
#         input_size: int,
#         freq: str,
#         prediction_length: int,
#         target_dim: int,
#         trainer: Trainer = Trainer(),
#         context_length: Optional[int] = None,
#         num_layers: int = 3,
#         num_cells: int = 256,
#         cell_type: str = "LSTM",
#         num_parallel_samples: int = 100,
#         dropout_rate: float = 0.1,
#         flow_type="RealNVP",
#         n_blocks=3,
#         hidden_size=100,
#         n_hidden=3,
#         conditioning_length: int = 200,
#         dequantize: bool = False,
#         scaling: bool = True,
#         pick_incomplete: bool = False,
#         lags_seq: Optional[List[int]] = None,
#         **kwargs,
#     ) -> None:
#         super().__init__(trainer=trainer, **kwargs)

#         self.freq = freq
#         self.context_length = (
#             context_length if context_length is not None else prediction_length
#         )

#         self.input_size = input_size
#         self.prediction_length = prediction_length
#         self.target_dim = target_dim
#         self.num_layers = num_layers
#         self.num_cells = num_cells
#         self.cell_type = cell_type
#         self.num_parallel_samples = num_parallel_samples
#         self.dropout_rate = dropout_rate

#         self.flow_type = flow_type
#         self.n_blocks = n_blocks
#         self.hidden_size = hidden_size
#         self.n_hidden = n_hidden
#         self.conditioning_length = conditioning_length
#         self.dequantize = dequantize

#         self.lags_seq = (
#             lags_seq
#             if lags_seq is not None
#             else lags_for_fourier_time_features_from_frequency(freq_str=freq)
#         )

#         # Time features are explicitly disabled
#         self.time_features = None

#         self.history_length = self.context_length + max(self.lags_seq)
#         self.pick_incomplete = pick_incomplete
#         self.scaling = scaling

#         self.train_sampler = ExpectedNumInstanceSampler(
#             num_instances=1.0,
#             min_past=0 if pick_incomplete else self.history_length,
#             min_future=prediction_length,
#         )

#         self.validation_sampler = ValidationSplitSampler(
#             min_past=0 if pick_incomplete else self.history_length,
#             min_future=prediction_length,
#         )

#     def create_transformation(self) -> Transformation:
#         return Chain(
#             [
#                 AsNumpyArray(
#                     field=FieldName.TARGET,
#                     expected_ndim=2,
#                 ),
#                 # Maps the target to (1, T) if the target data is unidimensional
#                 ExpandDimArray(
#                     field=FieldName.TARGET,
#                     axis=None,
#                 ),
#                 AddObservedValuesIndicator(
#                     target_field=FieldName.TARGET,
#                     output_field=FieldName.OBSERVED_VALUES,
#                 ),
#                 SetFieldIfNotPresent(field=FieldName.FEAT_STATIC_CAT, value=[0]),
#                 TargetDimIndicator(
#                     field_name="target_dimension_indicator",
#                     target_field=FieldName.TARGET,
#                 ),
#                 AsNumpyArray(field=FieldName.FEAT_STATIC_CAT, expected_ndim=1),
#             ]
#         )

#     def create_instance_splitter(self, mode: str):
#         assert mode in ["training", "validation", "test"]

#         instance_sampler = {
#             "training": self.train_sampler,
#             "validation": self.validation_sampler,
#             "test": TestSplitSampler(),
#         }[mode]

#         return InstanceSplitter(
#             target_field=FieldName.TARGET,
#             is_pad_field=FieldName.IS_PAD,
#             start_field=FieldName.START,
#             forecast_start_field=FieldName.FORECAST_START,
#             instance_sampler=instance_sampler,
#             past_length=self.history_length,
#             future_length=self.prediction_length,
#             time_series_fields=[
#                 FieldName.OBSERVED_VALUES,
#             ],
#         )

#     def create_training_network(self, device: torch.device) -> TempFlowTrainingNetwork:
#         return TempFlowTrainingNetwork(
#             input_size=self.input_size,
#             target_dim=self.target_dim,
#             num_layers=self.num_layers,
#             num_cells=self.num_cells,
#             cell_type=self.cell_type,
#             history_length=self.history_length,
#             context_length=self.context_length,
#             prediction_length=self.prediction_length,
#             dropout_rate=self.dropout_rate,
#             lags_seq=self.lags_seq,
#             scaling=self.scaling,
#             flow_type=self.flow_type,
#             n_blocks=self.n_blocks,
#             hidden_size=self.hidden_size,
#             n_hidden=self.n_hidden,
#             conditioning_length=self.conditioning_length,
#             dequantize=self.dequantize,
#         ).to(device)

#     def create_predictor(
#         self,
#         transformation: Transformation,
#         trained_network: TempFlowTrainingNetwork,
#         device: torch.device,
#     ) -> Predictor:
#         prediction_network = TempFlowPredictionNetwork(
#             input_size=self.input_size,
#             target_dim=self.target_dim,
#             num_layers=self.num_layers,
#             num_cells=self.num_cells,
#             cell_type=self.cell_type,
#             history_length=self.history_length,
#             context_length=self.context_length,
#             prediction_length=self.prediction_length,
#             dropout_rate=self.dropout_rate,
#             lags_seq=self.lags_seq,
#             scaling=self.scaling,
#             flow_type=self.flow_type,
#             n_blocks=self.n_blocks,
#             hidden_size=self.hidden_size,
#             n_hidden=self.n_hidden,
#             conditioning_length=self.conditioning_length,
#             dequantize=self.dequantize,
#             num_parallel_samples=self.num_parallel_samples,
#         ).to(device)

#         copy_parameters(trained_network, prediction_network)
#         input_names = get_module_forward_input_names(prediction_network)
#         prediction_splitter = self.create_instance_splitter("test")

#         return PyTorchPredictor(
#             input_transform=transformation + prediction_splitter,
#             input_names=input_names,
#             prediction_net=prediction_network,
#             batch_size=self.trainer.batch_size,
#             freq=self.freq,
#             prediction_length=self.prediction_length,
#             device=device,
#         )


