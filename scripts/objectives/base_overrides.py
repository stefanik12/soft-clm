from typing import Any
from typing import List, Union, Dict, Iterable, Optional

import torch
from adaptor.lang_module import LangModule
from adaptor.objectives.CLM import DataCollatorForCausalLM, CausalLanguageModeling
from adaptor.objectives.distillation import Distillation
from adaptor.objectives.objective_base import Objective
from torch.nn import CrossEntropyLoss
from transformers import DataCollatorForSeq2Seq, BatchEncoding


class ExperimentOverrides(Objective):

    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.collator = DataCollatorForCausalLM(self.tokenizer, self.compatible_head_model)

        # Add pad token to all models if using pythia
        if "pythia" in self.compatible_head_model.name_or_path:
            self.compatible_head_model.pad_token = "<|endoftext|>"
            self.tokenizer.pad_token = "<|endoftext|>"
            self.tokenizer.model_max_length = 1024

    def register_compatible_head_model(self,
                                       lang_module: LangModule,
                                       other_objective: Optional["Objective"] = None,
                                       objective_args_for_head_config: Optional[Dict[str, Any]] = None,
                                       preloaded_module: Optional[torch.nn.Module] = None) -> torch.nn.Module:

        head_module = super().register_compatible_head_model(lang_module, other_objective,
                                                             objective_args_for_head_config, preloaded_module)
        # assert hasattr(head_module, "prepare_decoder_input_ids_from_labels"), \
        #     "No head of the loaded LangModule is compatible with %s objective! " \
        #     "\nNote that the module compatible with " \
        #     "Sequence2SequenceMixin \nmust have `prepare_decoder_input_ids_from_labels` method, " \
        #     "see e.g. \ntransformers.BartModel." % self

        return head_module


class NewDataCollatorForCausalLM(DataCollatorForSeq2Seq):
    # TODO: this needs to be merged into adaptor when it starts working (after proper celebration)

    @staticmethod
    def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
        """
        Shift input ids one token to the right.
        From transformers.modeling_bart.
        """
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = decoder_start_token_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids

    def __call__(self,
                 features: List[Union[BatchEncoding, Dict[str, Iterable[Union[int, float]]]]],
                 return_tensors=None) -> BatchEncoding:
        """
        Custom DataCollator allowing to apply CausalLM also on models with fully-attended encoder.
        :param features: features to align
        :param return_tensors: Whether to return an encoding of tensors or lists.
        :return:
        """
        if return_tensors is None:
            return_tensors = self.return_tensors

        self.label_pad_token_id = -100
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        max_length = max([len(feature["input_ids"]) for feature in features])

        if labels is not None:
            max_tgt = max([len(feature["labels"]) for feature in features])
            max_length = max(max_length, max_tgt)
            # padding to max length of labels in batch
            # max_label_length = max(len(l) for l in labels)
            # padding to max length of labels and input_ids in batch
            max_label_length = max_length

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                feature["labels"] = (
                    feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                )
        num_features = len(features)
        out_features = self.tokenizer.pad(
            features,
            padding="max_length",
            max_length=max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids, if model requires it
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=out_features["input_ids"])
            out_features["decoder_input_ids"] = decoder_input_ids

            # encoder causal mask is full by default of translation models,
            # without it, model learns to just copy the input
            # with standard attention_mask, we rely on a resolution of AutoModelForCausalLM for CLM objectives
            causal_mask = torch.tril(torch.ones(max_length, max_length, dtype=torch.int32), diagonal=0)  # attended pos
            causal_mask = causal_mask.expand(num_features, max_length, max_length)  # for batch_size
            out_features["encoder_attention_mask"] = causal_mask

        bos_id = self.model.config.bos_token_id if self.model.config.bos_token_id is not None else 0
        pad_id = self.model.config.pad_token_id if self.model.config.pad_token_id is not None else 0

        # no shifting of the labels here: this happens in the corresponding loss fn
        labels = out_features["input_ids"].clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        out_features["labels"] = labels

        return out_features


class BaselineCLM(CausalLanguageModeling, ExperimentOverrides):

    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.collator = NewDataCollatorForCausalLM(self.tokenizer, self.compatible_head_model)

    def _compute_loss(self,
                      logit_outputs: torch.FloatTensor,
                      labels: torch.LongTensor,
                      inputs: Optional[Union[BatchEncoding, Dict[str, torch.Tensor]]] = None) -> torch.FloatTensor:
        """
        Causal language modeling, as implemented by GPT-2.

        :param inputs: Input encoding corresponding to given `logit_outputs` and `labels`.
        :param logit_outputs: Raw output of this objective's head.
        :param labels: Expected true labels of this objective.

        :return: a single-item torch tensor with registered grad_fn.
        """
        shift_logits = logit_outputs[..., :-1, :].contiguous()
        # Shifted labels
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return loss


class DistilledCLM(Distillation, BaselineCLM):
    pass

