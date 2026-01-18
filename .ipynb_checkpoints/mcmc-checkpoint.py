import torch
import sys
import torch.nn as nn

from base import Flow

# class MetropolisMCFlow(Flow):
#     def __init__(self, energy_model, nsteps=1, stepsize=0.01):
#         """ Stochastic Flow layer that simulates Metropolis Monte Carlo

#         """
#         super().__init__()
#         self.energy_model = energy_model
#         self.nsteps = nsteps
#         self.stepsize = stepsize
    
#     def _forward(self, x, **kwargs):
class MetropolisMCFlow(nn.Module):
    def __init__(self, nsteps=1, stepsize=0.01):
        super().__init__()
        self.nsteps = nsteps
        self.stepsize = stepsize

    def _forward(self, x, energy_model):

        """ Run a stochastic trajectory forward 
        
        Parameters
        ----------
        x : PyTorch Tensor
            Batch of input configurations
        
        Returns
        -------
        x' : PyTorch Tensor
            Transformed configurations
        dW : PyTorch Tensor
            Nonequilibrium work done, always 0 for this process
            
        """
        #x_E = x.squeeze(1) 
        #print("x (input to energy_model):", x_E.shape)
        # print("energy_model(x):", energy_model(x).shape)
        #print("[MCMC] x shape:", x.shape)
        # if hasattr(energy_model, "energy_model_1") and hasattr(energy_model.energy_model_1, "target_pred"):
        #     print("[MCMC] self.target_pred shape:", energy_model.energy_model_1.target_pred.shape)
        #     print("[MCMC] self.target_pred.unsqueeze(1) shape:", energy_model.energy_model_1.target_pred.unsqueeze(1).shape)


        # E0 = self.energy_model.energy(x)
        E0 = energy_model(x)
        E = E0

        for i in range(self.nsteps):
            # proposal step
            #dx = self.stepsize * torch.zeros_like(x).normal_()
            dx = self.stepsize * torch.randn_like(x)  # same as above, more readable
            xprop = x + dx
            #xprop_E = xprop.squeeze(1) 
            #print("xprop:", xprop.shape)
            #Eprop = self.energy_model.energy(xprop)
            Eprop = energy_model(xprop)
            
            # acceptance step
            #acc = (torch.rand(x.shape[0], 1) < torch.exp(-(Eprop - E))).float()  # selection variable: 0 or 1.
            #print("[MCMC] x shape:", x.shape)
            # print("[MCMC] E shape:", E.shape)
            # print("[MCMC] Eprop shape:", Eprop.shape)

            acc = (torch.rand(x.shape, device=x.device) < torch.exp(-(Eprop -E))).float()
            #print("acc:", acc.shape)
            #acc = acc.expand_as(x)
            accept_rate = acc.float().mean().item()
            #print(f"[MCMC] Accept Rate: {accept_rate:.3f}")
            # print("[MCMC] acc shape:", acc.shape)
            # print("[MCMC] x shape (before update):", x.shape)
            # print("[MCMC] xprop shape:", xprop.shape)

            x = (1 - acc) * x + acc * xprop
            E = (1 - acc) * E + acc * Eprop

        # Work is energy difference
        dW = E - E0
        
        return x, dW

    def _inverse(self, x, **kwargs):
        """ Same as forward """
        return self._forward(x, **kwargs)
    