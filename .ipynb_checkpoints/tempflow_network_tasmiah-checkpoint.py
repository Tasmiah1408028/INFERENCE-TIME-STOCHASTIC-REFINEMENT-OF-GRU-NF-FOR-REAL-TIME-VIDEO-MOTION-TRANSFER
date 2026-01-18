from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from gluonts.core.component import validated

from pts.model import weighted_average
from pts.modules import RealNVP, MAF, FlowOutput, MeanScaler, NOPScaler

#without lags seq
class TempFlowTrainingNetwork(nn.Module):
    @validated()
    def __init__(
        self,
        input_size: int,
        num_layers: int,
        num_cells: int,
        cell_type: str,
        history_length: int,
        context_length: int,
        prediction_length: int,
        dropout_rate: float,
        lags_seq: List[int],
        target_dim: int,
        conditioning_length: int,
        flow_type: str,
        n_blocks: int,
        hidden_size: int,
        n_hidden: int,
        dequantize: bool,
        cardinality: List[int] = [1],
        embedding_dimension: int = 1,
        scaling: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.target_dim = target_dim
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.history_length = history_length
        self.scaling = scaling

        # Remove lags_seq
        self.lags_seq = []

        self.cell_type = cell_type
        rnn_cls = {"LSTM": nn.LSTM, "GRU": nn.GRU}[cell_type]
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=num_cells,
            num_layers=num_layers,
            dropout=dropout_rate,
            batch_first=True,
        )

        flow_cls = {
            "RealNVP": RealNVP,
            "MAF": MAF,
        }[flow_type]
        self.flow = flow_cls(
            input_size=target_dim,
            n_blocks=n_blocks,
            n_hidden=n_hidden,
            hidden_size=hidden_size,
            cond_label_size=conditioning_length,
        )
        self.dequantize = dequantize

        self.distr_output = FlowOutput(
            self.flow, input_size=target_dim, cond_size=conditioning_length
        )

        self.proj_dist_args = self.distr_output.get_args_proj(num_cells)

        # Remove embedding layer
        # self.embed_dim = 1
        # self.embed = nn.Embedding(num_embeddings=self.target_dim, embedding_dim=self.embed_dim)

        if self.scaling:
            self.scaler = MeanScaler(keepdim=True)
        else:
            self.scaler = NOPScaler(keepdim=True)

    def unroll(
        self,
        past_target_cdf: torch.Tensor,
        scale: torch.Tensor,
        unroll_length: int,
        begin_state: Optional[Union[List[torch.Tensor], torch.Tensor]] = None,
    ) -> Tuple[
        torch.Tensor,
        Union[List[torch.Tensor], torch.Tensor],
        torch.Tensor,
        torch.Tensor,
    ]:
        # Use only the raw input without lags, embeddings, or time features
        inputs = past_target_cdf[:, -unroll_length:, :]

        # Print input shape before GRU
        # print(f"Final input shape before GRU: {inputs.shape}")

        # Unroll encoder
        outputs, state = self.rnn(inputs, begin_state)

        # Return outputs, state, scale, and inputs
        return outputs, state, scale, inputs

    def unroll_encoder(
        self,
        past_target_cdf: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_is_pad: torch.Tensor,
        future_target_cdf: Optional[torch.Tensor],
        target_dimension_indicator: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        Union[List[torch.Tensor], torch.Tensor],
        torch.Tensor,
        torch.Tensor,
    ]:

        past_observed_values = torch.min(
            past_observed_values, 1 - past_is_pad.unsqueeze(-1)
        )

        if future_target_cdf is None:
            sequence = past_target_cdf
            subsequences_length = self.context_length
        else:
            sequence = torch.cat((past_target_cdf, future_target_cdf), dim=1)
            subsequences_length = self.context_length + self.prediction_length
        # sequence = past_target_cdf
        # subsequences_length = self.context_length

        # Scale is computed on the context length last units of the past target
        _, scale = self.scaler(
            past_target_cdf[:, -self.context_length :, ...],
            past_observed_values[:, -self.context_length :, ...],
        )

        outputs, states, scale, inputs = self.unroll(
            past_target_cdf=sequence[:, :subsequences_length, :],
            scale=scale,
            unroll_length=subsequences_length,
            begin_state=None,
        )

        return outputs, states, scale, inputs

    def distr_args(self, rnn_outputs: torch.Tensor):
        (distr_args,) = self.proj_dist_args(rnn_outputs)
        return distr_args

    def forward(
        self,
        target_dimension_indicator: torch.Tensor,
        past_target_cdf: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_is_pad: torch.Tensor,
        future_target_cdf: torch.Tensor,
        future_observed_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:

        rnn_outputs, _, scale, _ = self.unroll_encoder(
            past_target_cdf=past_target_cdf,
            past_observed_values=past_observed_values,
            past_is_pad=past_is_pad,
            future_target_cdf=future_target_cdf,
            target_dimension_indicator=target_dimension_indicator,
        )

        target = torch.cat(
            (past_target_cdf[:, -self.context_length :, ...], future_target_cdf),
            dim=1,
        )
        # target = past_target_cdf[:, -self.context_length :, ...]
        # target_2 = future_target_cdf
        distr_args = self.distr_args(rnn_outputs=rnn_outputs)
        if self.scaling:
            self.flow.scale = scale

        if self.dequantize:
            target += torch.rand_like(target)
        likelihoods = -self.flow.log_prob(target, distr_args).unsqueeze(-1)

        past_observed_values = torch.min(
            past_observed_values, 1 - past_is_pad.unsqueeze(-1)
        )

        observed_values = torch.cat(
            (
                past_observed_values[:, -self.context_length :, ...],
                future_observed_values,
            ),
            dim=1,
        )
        #observed_values = past_observed_values[:, -self.context_length :, ...]
        loss_weights, _ = observed_values.min(dim=-1, keepdim=True)
        # Compute likelihood-based loss(new)
        likelihood_loss = weighted_average(likelihoods, weights=loss_weights, dim=1)


        # Combine likelihood loss and MSE loss
        total_loss = likelihood_loss.mean() #+ mse_loss

        #return total_loss, likelihoods, distr_args
        #loss = weighted_average(likelihoods, weights=loss_weights, dim=1)

        return total_loss, likelihoods, distr_args #loss.mean()


class TempFlowPredictionNetwork(TempFlowTrainingNetwork):
    def __init__(self, num_parallel_samples: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_parallel_samples = num_parallel_samples

    def sampling_decoder(
        self,
        past_target_cdf: torch.Tensor,
        target_dimension_indicator: torch.Tensor,
        scale: torch.Tensor,
        begin_states: Union[List[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:

        def repeat(tensor, dim=0):
            return tensor.repeat_interleave(repeats=self.num_parallel_samples, dim=dim)

        repeated_past_target_cdf = past_target_cdf
        repeated_scale = scale
        if self.scaling:
            self.flow.scale = repeated_scale
        repeated_target_dimension_indicator = target_dimension_indicator

        # if self.cell_type == "LSTM":
        #     repeated_states = [repeat(s, dim=1) for s in begin_states]
        # else:
        repeated_states = begin_states

        future_samples = []
        #_, state = self.rnn(repeated_past_target_cdf,repeated_states)
        for k in range(self.prediction_length):
            rnn_outputs, repeated_states, _, _ = self.unroll(
                past_target_cdf=repeated_past_target_cdf,
                scale=repeated_scale,
                unroll_length=1,
                begin_state=repeated_states,
            )

            distr_args = self.distr_args(rnn_outputs=rnn_outputs)
            new_samples = torch.stack([self.flow.sample(cond=distr_args) for _ in range(self.num_parallel_samples)])
            mean_sample = new_samples.mean(dim=0)
            future_samples.append(new_samples)
            #mean_new_sample = new_samples.view(-1, self.num_parallel_samples, 1, self.target_dim).mean(dim=1)
            repeated_past_target_cdf = torch.cat(
                (repeated_past_target_cdf, mean_sample), dim=1
            )

        samples = torch.cat(future_samples, dim=1)

        return samples.reshape(
            (
                -1,
                self.num_parallel_samples,
                self.prediction_length,
                self.target_dim,
            )
        )

    def forward(
        self,
        target_dimension_indicator: torch.Tensor,
        past_target_cdf: torch.Tensor,
        past_observed_values: torch.Tensor,
        past_is_pad: torch.Tensor,
    ) -> torch.Tensor:

        _, begin_states, scale, _ = self.unroll_encoder(
            past_target_cdf=past_target_cdf,
            past_observed_values=past_observed_values,
            past_is_pad=past_is_pad,
            future_target_cdf=None,
            target_dimension_indicator=target_dimension_indicator,
        )

        return self.sampling_decoder(
            past_target_cdf=past_target_cdf,
            target_dimension_indicator=target_dimension_indicator,
            scale=scale,
            begin_states=begin_states,
        )
