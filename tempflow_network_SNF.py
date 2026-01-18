from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import math
from statsmodels.tsa.api import VAR

from gluonts.core.component import validated

from pts.model import weighted_average
from pts.modules import RealNVP, MAF, FlowOutput, MeanScaler, NOPScaler
import sys
from flows_SNF import SRealNVP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        mcmc_steps: int = 2,
        mcmc_eps: float = 0.35,
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
            "SRealNVP": SRealNVP,
            "MAF": MAF,
        }[flow_type]
        # Build the flow instance depending on type
        if flow_type == "SRealNVP":
            self.flow = flow_cls(
                input_size=target_dim,
                n_blocks=n_blocks,
                n_hidden=n_hidden,
                hidden_size=hidden_size,
                cond_label_size=conditioning_length,
                mcmc_steps=mcmc_steps,
                mcmc_eps=mcmc_eps,
            )
        else:
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

        return total_loss, likelihoods, distr_args #loss.mean()


class TempFlowPredictionNetwork(TempFlowTrainingNetwork):
    def __init__(self, num_parallel_samples: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_parallel_samples = num_parallel_samples
        self.debug_predicted_targets = []
        # Linear head to map GRU output to keypoint prediction
        self.gru_predictor = nn.Sequential(
            nn.Linear(kwargs["num_cells"], self.target_dim)
        )

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
        repeated_states = begin_states

        future_samples = []
        # mu = torch.tensor([2.0, -2.0, 1.0, 1.0, -1.0], device=device)  # shape [5]
        # sigma = 0.5

        for k in range(self.prediction_length):
            rnn_outputs, repeated_states, _, _ = self.unroll(
                past_target_cdf=repeated_past_target_cdf,
                scale=repeated_scale,
                unroll_length=1,
                begin_state=repeated_states,
            )

            distr_args = self.distr_args(rnn_outputs=rnn_outputs)
            # class VAREnergyModel(nn.Module):
            #     def __init__(self, past_sequence):
            #         super().__init__()
            #         self.past_sequence = past_sequence.cpu().numpy()  # shape: [batch, context_len, dim]
            
            #         # Fit a VAR model per batch element
            #         self.predictions = []
            #         for i in range(self.past_sequence.shape[0]):
            #             series = self.past_sequence[i]  # [context_len, dim]
            #             model = VAR(series)
            #             fitted = model.fit(maxlags=1)  # You can tune maxlags
            #             pred = fitted.forecast(series[-1:], steps=1)
            #             self.predictions.append(torch.tensor(pred[0], dtype=torch.float32))
            
            #         self.predictions = torch.stack(self.predictions).to(past_sequence.device)  # [batch, dim]
            
            #     def forward(self, y, context=None):
            #         y = y.squeeze(1)  # [batch, dim]
            #         return ((self.predictions - y) ** 2).unsqueeze(1)  # [batch, 1, dim]
            
            #     def __call__(self, y, context=None):
            #         return self.forward(y, context)
            # context_input = repeated_past_target_cdf[:, -10:, :]  # [batch, context_len, dim]
            # energy_model = VAREnergyModel(context_input)

            #context_input = repeated_past_target_cdf[:, -1, :]

            # ---- Energy model components ----
            # Predict GRU keypoint target
            predicted_target = self.gru_predictor(rnn_outputs[:, -1, :])  # [batch, target_dim]
            # self.debug_predicted_targets.append(predicted_target.detach().cpu())
            # out = self.gru_predictor(rnn_outputs[:, -1, :])  # [batch, 2 * target_dim]
            # predicted_target1 = out[:, :self.target_dim]   # [batch, target_dim]
            # predicted_target2 = out[:, self.target_dim:]   # [batch, target_dim]

            #Define energy model wrapper
            class GRUEnergyModel(nn.Module):
                def __init__(self, target_pred):
                    super().__init__()
                    self.target_pred = target_pred  # [batch, target_dim]

                def forward(self, y, _context):
                    # print("self.target_pred shape:", self.target_pred.shape)
                    # print("y shape:", y.shape)
                    #return torch.sum((self.target_pred - y.squeeze(1)) ** 2, dim=-1, keepdim=True).unsqueeze(1)
                    return ((self.target_pred - y.squeeze(1)) ** 2).unsqueeze(1)  # shape [64, 1, 5]
                def __call__(self, y, context=None):
                    return self.forward(y, context)

            energy_model = GRUEnergyModel(predicted_target)
            # class GRUEnergyModel(nn.Module):
            #     def __init__(self, target_pred1, target_pred2):
            #         super().__init__()
            #         self.target_pred1 = target_pred1  # [batch, target_dim]
            #         self.target_pred2 = target_pred2  # [batch, target_dim]
            
            #     def forward(self, y, _context):
            #         y = y.squeeze(1)  # [batch, target_dim]
                    
            #         energy1 = torch.sum((self.target_pred1 - y) ** 2, dim=-1, keepdim=True)  # [batch, 1]
            #         energy2 = torch.sum((self.target_pred2 - y) ** 2, dim=-1, keepdim=True)  # [batch, 1]
                    
            #         combined_energy = -torch.logsumexp(torch.cat([-energy1, -energy2], dim=-1), dim=-1, keepdim=True)  # [batch, 1]
                    
            #         return combined_energy.unsqueeze(1)  # [batch, 1, 1]
            
            #     def __call__(self, y, context=None):
            #         return self.forward(y, context)
            # energy_model = GRUEnergyModel(predicted_target1, predicted_target2)

            # class MixtureEnergyModel(nn.Module):
            #     def __init__(self, mu, sigma):
            #         super().__init__()
            #         self.mu = nn.Parameter(mu, requires_grad=False)  # [dim]
            #         self.sigma = sigma
                
            #     def forward(self, y, context):
            #         """
            #         y: [batch, dim]
            #         context: [batch, dim]
            #         """
            #         y = y.squeeze(1)
                        
            #         batch_size = y.shape[0]
            #         cov = (self.sigma ** 2) * torch.eye(len(self.mu), device=y.device).unsqueeze(0).expand(batch_size, -1, -1)

            #         #cov = (self.sigma ** 2) * torch.eye(len(self.mu), device=y.device)
                    
            #         mean1 = context + self.mu
            #         mean2 = context - self.mu
                
            #         mvn1 = torch.distributions.MultivariateNormal(mean1, covariance_matrix=cov)
            #         mvn2 = torch.distributions.MultivariateNormal(mean2, covariance_matrix=cov)
                
            #         logp1 = mvn1.log_prob(y)
            #         logp2 = mvn2.log_prob(y)
                
            #         log_mix = torch.logsumexp(torch.stack([logp1, logp2], dim=0), dim=0) - torch.log(torch.tensor(2.0, device=y.device))
            #         return -log_mix.unsqueeze(-1).unsqueeze(-1)  # shape [batch, 1]
                
            #     def __call__(self, y, context):
            #         return self.forward(y, context)

            # class MixtureEnergyModel(nn.Module):
            #     def __init__(self, mu, sigma):
            #         super().__init__()
            #         self.mu = nn.Parameter(mu, requires_grad=False)  # [dim]
            #         self.sigma = sigma
            
            #     def forward(self, y, context):
            #         """
            #         y: [batch, 1, dim] or [batch, dim]
            #         context: [batch, dim]
            #         Returns: [batch, 1, dim] energy per dimension
            #         """
            #         y = y.squeeze(1)  # [batch, dim]
            
            #         mean1 = context + self.mu  # [batch, dim]
            #         mean2 = context - self.mu  # [batch, dim]
            
            #         # Per-dimension log-prob under each Gaussian (assume diagonal covariance)
            #         logp1 = -0.5 * ((y - mean1) ** 2 / self.sigma ** 2 + math.log(2 * math.pi * self.sigma ** 2))  # [batch, dim]
            #         logp2 = -0.5 * ((y - mean2) ** 2 / self.sigma ** 2 + math.log(2 * math.pi * self.sigma ** 2))  # [batch, dim]
            
            #         # Log-sum-exp across the two mixture components, per dimension
            #         log_mix = torch.logsumexp(torch.stack([logp1, logp2], dim=0), dim=0) - math.log(2.0)  # [batch, dim]
            
            #         return -log_mix.unsqueeze(1)  # [batch, 1, dim]
            
            #     def __call__(self, y, context):
            #         return self.forward(y, context)


            # energy_model = MixtureEnergyModel(mu,sigma)

            # ---- Draw samples using SNF + MCMC + energy ----
            new_samples = torch.stack([
                self.flow.sample_with_energy(cond=distr_args, energy_model=energy_model, context=None)  #.squeeze(1)
                for _ in range(self.num_parallel_samples)
            ])

            mean_sample = new_samples.mean(dim=0)
            future_samples.append(new_samples)
            # print("repeated_past_target_cdf shape:", repeated_past_target_cdf.shape)
            # print("mean_sample shape:", mean_sample.shape)

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

