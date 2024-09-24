from typing import Optional, Dict, Union

import torch
from adaptor.objectives.CLM import CausalLanguageModeling
from transformers import DataCollatorForSeq2Seq, BatchEncoding

from scripts.objectives.base_overrides import ExperimentOverrides


class SoftCLM(CausalLanguageModeling, ExperimentOverrides):

    def __init__(self, *args, **kwargs):
        """
        Refer to the documentation of superclass.
        """
        # adjust only default max_samples_per_*log, since generative evaluation is much slower
        # but stick to user selection, if there is any
        if "max_samples_per_log" not in kwargs:
            kwargs["max_samples_per_log"] = 200
        if "max_samples_per_eval_log" not in kwargs:
            kwargs["max_samples_per_eval_log"] = 1000

        super().__init__(*args, **kwargs)

        # if this is translation objective, tokenization of source and target might vary (can include lang_token_id)
        # if it does not, this will just set unused attribute of tokenizer
        self.collator = DataCollatorForSeq2Seq(self.tokenizer, self.compatible_head_model, pad_to_multiple_of=8)

        # TODO: pre-compute the token similarities in a training corpus

    def _compute_loss(self,
                      lm_logit_outputs: torch.FloatTensor,
                      labels: torch.LongTensor,
                      inputs: Optional[Union[BatchEncoding, Dict[str, torch.Tensor]]] = None,
                      ignore_index: int = -100) -> torch.FloatTensor:
        """
        Computes sequence2sequence loss
        :param logit_outputs: Raw outputs of language modeling head model
        :param labels: Token ids of expected outputs.
        :return: Single value of the loss, with grad_fn.
        """
        # note that currently we do not ignore padding from the loss, which might be desirable
        # - we have seen this to eliminate repetitive generations at some cases
        loss_fct = torch.nn.CrossEntropyLoss()

        logits_f = lm_logit_outputs.flatten(end_dim=1)
        labels_f = labels.flatten(end_dim=1)

        # TODO: construct training labels from the similarity of all X current_label


        if lm_logit_outputs.shape == labels.shape:
            # for non-discrete targets, torch loss will not ignore the ignore_index targets,
            # so we exclude them manually
            ignore_idx = (labels_f == ignore_index).all(-1)
            logits_f = logits_f[~ignore_idx]
            labels_f = labels_f[~ignore_idx]

        lm_loss = loss_fct(logits_f, labels_f)

        return lm_loss
