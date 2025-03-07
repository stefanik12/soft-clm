import inspect
from collections import deque
from typing import Any
from typing import List, Union, Dict, Iterable, Optional

import torch
from adaptor.lang_module import LangModule
from adaptor.objectives.CLM import DataCollatorForCausalLM, CausalLanguageModeling
from adaptor.objectives.distillation import Distillation
from adaptor.objectives.objective_base import Objective
from torch import log_softmax, softmax
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

        # TODO: caution about the new collator
        # self.collator = NewDataCollatorForCausalLM(self.tokenizer, self.compatible_head_model)
        self.collator = DataCollatorForSeq2Seq(self.tokenizer, self.compatible_head_model, pad_to_multiple_of=8)

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

    def __init__(self, *args, force_true_tokens: bool = False,
                 force_false_tokens: bool = False,
                 rho_token_selection_ratio: float = 1.0,
                 rho_threshold_batch_window: int = 512,
                 **kwargs) -> None:
        self.force_true_tokens = force_true_tokens
        self.force_false_tokens = force_false_tokens
        self.rho_token_selection_ratio = rho_token_selection_ratio
        super().__init__(*args, **kwargs)
        self.running_rho_threshold_window = deque([], maxlen=rho_threshold_batch_window // self.batch_size)

    def _compute_loss(self,
                      student_logits: torch.FloatTensor,
                      labels: torch.LongTensor,
                      inputs: Optional[Union[BatchEncoding, Dict[str, torch.Tensor]]] = None) -> torch.FloatTensor:
        assert inputs is not None, "Distillation loss requires model inputs to be passed"

        # output logits' loss
        ce_loss = CrossEntropyLoss(reduction="none")

        teacher_inputs = inspect.getfullargspec(self.teacher_model.forward).args

        device = student_logits.device
        if self.teacher_model.device != device:
            self.teacher_model = self.teacher_model.to(device)

        if self.force_true_tokens and self.force_false_tokens and self.rho_token_selection_ratio == 1.0:
            # baseline training: if we don't use either true or false tokens of the teacher, or rho,
            # we do not need to do inference with the teacher model at all!
            teacher_logits = torch.zeros_like(student_logits)
            teacher_probs = torch.zeros_like(student_logits)
        else:
            with torch.no_grad():
                teacher_outputs = self.teacher_model(**{k: v.to(device) for k, v in inputs.items() if k in teacher_inputs})
                teacher_logits = teacher_outputs.logits
                teacher_probs = softmax(teacher_logits, dim=-1)

        # TODO: with force_true_tokens, consider restrict_loss_to_mask!
        if self.restrict_loss_to_mask:
            # pick only the predictions of tokens on the attended positions (i.e. ignore the others)
            attn_mask_reshaped = inputs["attention_mask"].unsqueeze(-1).expand_as(student_logits).bool()

            student_logits_flat = torch.masked_select(student_logits, attn_mask_reshaped)
            student_logits_unbatched = student_logits_flat.reshape(-1, student_logits.shape[-1])

            teacher_logits_flat = torch.masked_select(teacher_logits, attn_mask_reshaped)
            teacher_logits_unbatched = teacher_logits_flat.reshape(-1, teacher_logits.shape[-1])

            # we flatten the batch, to get the class scores & probabilities to the 2nd dimension
            teacher_probs_flat = torch.masked_select(teacher_probs, attn_mask_reshaped)
            teacher_probs_unbatched = teacher_probs_flat.reshape(-1, teacher_logits.shape[-1])

            # new input_ids should not be used for inference, so we keep them separate
            labels_unbatched = torch.masked_select(inputs["labels"], inputs["attention_mask"].bool())
            labels_shifted = torch.masked_select(inputs["labels"][..., 1:], inputs["attention_mask"][..., :-1].bool())
        else:
            # we flatten the batch, to get the class scores & probabilities to the 2nd dimension
            student_logits_unbatched = student_logits.flatten(end_dim=1)
            teacher_logits_unbatched = teacher_logits.flatten(end_dim=1)
            teacher_probs_unbatched = teacher_probs.flatten(end_dim=1)
            labels_unbatched = inputs["labels"].flatten()
            labels_shifted = inputs["labels"][..., 1:].flatten()

        if self.force_true_tokens:
            # set the probabilities of all true tokens from the reference to one
            ind0 = torch.arange(labels_shifted.numel(), device=device)
            # teacher_probs_unbatched = teacher_probs.flatten(end_dim=1)
            teacher_probs_unbatched[ind0, labels_shifted] = 1.  # TODO check: error by one?
            # teacher_probs = teacher_probs_unbatched.reshape(teacher_probs.shape)

        if self.force_false_tokens:
            ind0 = torch.arange(labels_shifted.numel(), device=device)
            # teacher_probs_unbatched = teacher_probs.flatten(end_dim=1)
            zeroed_teacher_probs = torch.zeros_like(teacher_probs_unbatched, device=device)
            zeroed_teacher_probs[ind0, labels_shifted] = teacher_probs_unbatched[ind0, labels_shifted]
            # teacher_probs = zeroed_teacher_probs.reshape(teacher_probs.shape)
            teacher_probs_unbatched = zeroed_teacher_probs
        # TODO: dropped log_softmax from distil_loss: must be tested and then merged to Adaptor's distillation!
        distil_loss = ce_loss(log_softmax(student_logits_unbatched / self.temperature, dim=-1),
                              teacher_probs_unbatched / self.temperature) * (self.temperature ** 2)
        distil_loss = self.logits_ce_loss_weight * distil_loss

        if self.rho_token_selection_ratio != 1.0:
            # done: a scale of batch to compute threshold seems to make a large difference
            #  consider aggregating the rho on a running basis, perhaps to the size of common pre-training batch_sizes

            # TODO: for the perfect reproducibility of the actual rho method, check that
            #  with force_true_tokens==True and force_false_tokens==True, we perfectly match the loss of the baseline

            teacher_ce_loss = ce_loss(teacher_logits_unbatched[:labels_shifted.shape[0]], labels_shifted)
            student_ce_loss = ce_loss(student_logits_unbatched[:labels_shifted.shape[0]], labels_shifted)

            losses_delta = student_ce_loss - teacher_ce_loss
            batch_threshold = torch.quantile(losses_delta, 1 - self.rho_token_selection_ratio)
            self.running_rho_threshold_window.append(batch_threshold)

            # done: take a look at the tokens that are included/excluded -- do they match expected aleatoric tokens?
            current_threshold = sum(self.running_rho_threshold_window) / len(self.running_rho_threshold_window)
            distil_loss = distil_loss[:labels_shifted.shape[0]][losses_delta >= current_threshold]

        distil_loss = distil_loss.mean()

        # end output logits' loss

        if self.add_hidden_states_loss:
            # hidden states loss
            # second inference is needed since the objectives' interface provides only output logits (not other outputs)
            # currently, this does not consider restrict_loss_to_mask (but maybe we don't want that)
            student_inputs = inspect.getfullargspec(self.compatible_head_model.forward).args
            student_outputs = self.compatible_head_model(**{k: v for k, v in inputs.items() if k in student_inputs})

            hidden_loss = self._hidden_states_loss(student_outputs, teacher_outputs, inputs["attention_mask"])
            hidden_loss_scaled = self.hidden_cossim_loss_weight * hidden_loss

            distil_loss = distil_loss + hidden_loss_scaled

        return distil_loss
