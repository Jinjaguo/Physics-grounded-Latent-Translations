"""Exact non-bias-corrected exponential moving average for trainable parameters."""
from __future__ import annotations
import torch
class ParameterEMA:
    """Maintain an EMA shadow for every trainable parameter, excluding frozen tensors."""
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay=float(decay); self.shadow={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}; self.updates=0
    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for n,p in model.named_parameters():
            if n in self.shadow: self.shadow[n].mul_(self.decay).add_(p.detach(),alpha=1-self.decay)
        self.updates+=1
    def state_dict(self): return {'decay':self.decay,'updates':self.updates,'shadow':{k:v.clone() for k,v in self.shadow.items()}}
    def load_state_dict(self, state):
        self.decay=float(state['decay']); self.updates=int(state['updates'])
        self.shadow={k:v.detach().clone() for k,v in state['shadow'].items()}
    def store(self, model):
        self._stored={n:p.detach().clone() for n,p in model.named_parameters() if n in self.shadow}
    @torch.no_grad()
    def restore(self, model):
        if not hasattr(self,'_stored'): raise RuntimeError('No raw parameters have been stored')
        for n,p in model.named_parameters():
            if n in self._stored: p.copy_(self._stored[n])
        del self._stored
    @torch.no_grad()
    def copy_to(self, model):
        for n,p in model.named_parameters():
            if n in self.shadow: p.copy_(self.shadow[n])
