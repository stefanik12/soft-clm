import inspect
from typing import Type, Union

import torch


def norm_attention_from_existing_module(ExistingCls: Type[torch.nn.Module]) -> "NormAttention":

    assert hasattr(ExistingCls, "forward") and inspect.getfullargspec(ExistingCls.forward).args[1] == "hidden_states", \
        "NormAttention override of Attention expects the first argument of forward() to be hidden_states"

    class NormAttention(ExistingCls, torch.nn.Module):

        def __init__(self, *args, min_scale: float = 3., **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.min_scale = min_scale

        def forward(self, hidden_states: torch.FloatTensor, *args, **kwargs):
            # NormAttention rewritten from
            # https://github.com/boschresearch/eurekaMoments/blob/cd424d01b54dd0e5a41091e1851fb2e464b63be5/vit.py#L234

            # hidden_states /= self.scale  # ?? This just reverts multiplication of (q @ k) * scale from earlier
            # TODO: maybe also try norm_softmax variant:
            #  https://github.com/boschresearch/eurekaMoments/blob/cd424d01b54dd0e5a41091e1851fb2e464b63be5/vit.py#L240C40-L240C52
            s = torch.std(hidden_states, dim=-1, keepdim=True)
            hidden_states = (hidden_states / torch.minimum(torch.ones_like(s, device=hidden_states.device)
                                                           / self.min_scale, s)).softmax(dim=-1)
            return super().forward(hidden_states, *args, **kwargs)

    return NormAttention
