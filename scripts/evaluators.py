from typing import List, Dict, Any

import torch
from adaptor.evaluators.evaluator_base import EvaluatorBase
from adaptor.utils import AdaptationDataset, Head
from transformers import PreTrainedTokenizer

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM


class LMHarnessEvaluator(EvaluatorBase):

    compatible_heads = [Head.CLM]
    smaller_is_better = False

    def __init__(self,
                 decides_convergence: bool = False,
                 debug_mode: bool = False,
                 **evaluator_kwargs):
        super().__init__(decides_convergence)
        assert "tasks" in evaluator_kwargs, "LMHarnessEvaluator requires a list of tasks for evaluation."
        self.evaluator_kwargs = evaluator_kwargs
        if debug_mode:
            self.evaluator_kwargs["debug_on_first_subtask"] = True

    def __call__(self, model: torch.nn.Module,
                 tokenizer: PreTrainedTokenizer,
                 dataset: AdaptationDataset) -> Dict[str, float]:

        wrapped_model = HFLM(model, backend="causal", tokenizer=tokenizer, dtype=model.dtype,
                             batch_size=self.evaluator_kwargs.get("batch_size", 1))

        results = evaluator.simple_evaluate(model=wrapped_model, device=model.device, **self.evaluator_kwargs)

        # model.__class__ = orig_cls

        return {k: v['f1,none'] if 'acc,none' not in v else v['acc,none'] for k, v in results["results"].items()}

    def __str__(self) -> str:
        return "Acc"
