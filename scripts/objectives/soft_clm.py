import itertools
from typing import Optional, Dict, Union, Iterable

import torch
import torch.nn.functional as F
from adaptor.objectives.CLM import CausalLanguageModeling
from adaptor.objectives.distillation import Distillation
from tqdm import tqdm
from transformers import DataCollatorForSeq2Seq, BatchEncoding, PreTrainedModel, PreTrainedTokenizer

from objectives.base_overrides import ExperimentOverrides


class SoftCLM(CausalLanguageModeling, Distillation, ExperimentOverrides):

    def __init__(self, *args, normalize_similarities: bool = True, similarities_weight: float = 1.0, **kwargs):
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

        self.dtype = torch.float

        # if this is translation objective, tokenization of source and target might vary (can include lang_token_id)
        # if it does not, this will just set unused attribute of tokenizer
        self.collator = DataCollatorForSeq2Seq(self.tokenizer, self.compatible_head_model, pad_to_multiple_of=8)

        with torch.no_grad():
            eps = 1e-15
            embeddings = self._compute_embeddings(self._per_split_iterators("train")[0],
                                                  self.dataset_length["train"],
                                                  self.tokenizer,
                                                  self.teacher_model,
                                                  self.batch_size,
                                                  self.dtype,
                                                  init_eps=eps).to("cpu")
        unseen_embeddings_idx = embeddings.sum(1) < 1e-6  # TODO: parametrize eps
        mean_seen_embedding = embeddings[~unseen_embeddings_idx].mean(0)
        embeddings[unseen_embeddings_idx] = mean_seen_embedding  # should not affect the normalization

        self.similarities = (F.normalize(embeddings) @ F.normalize(embeddings).T).contiguous()  # normalized pairwise similarities

        # set similarity to/from unseen tokens to mean -> for similarities normalization
        mean_seen_similarity = self.similarities[~unseen_embeddings_idx][:, ~unseen_embeddings_idx].mean()
        self.similarities[unseen_embeddings_idx] = mean_seen_similarity
        self.similarities[:, unseen_embeddings_idx] = mean_seen_similarity

        # print(unseen_tokens_similarities.max(1))
        if normalize_similarities:
            mins = self.similarities.quantile(0.025, dim=1)
            # mins = topk_similarities_vals.min(dim=1)[0]
            maxs = 1
            # self.similarities -= self.similarities.min(1)[0]
            # self.similarities /= self.similarities.max(1)[0]
            self.similarities -= mins
            self.similarities /= (maxs - mins)  # upper-bound by the self-similarity (always equal to one)
        # TODO: make sure that after normalization, semantically-equivalent tokens are still largely similar:
        #  -> see self.similarities[16831, 6968]
        #  supposedly, without normalization, this objective can not work in a standalone

        unseen_target_sims = torch.eye(self.similarities.shape[0], device=self.similarities.device)[unseen_embeddings_idx]

        # TODO: check that both unseen and seen tokens have high true-token probability
        self.similarities[unseen_embeddings_idx] = unseen_target_sims  # unseen tokens -> all get one-hot sims
        self.similarities[~unseen_embeddings_idx][:, unseen_embeddings_idx] = 0  # seen tokens -> unseen tokens get zero
        self.similarities[self.similarities < 0] = 0
        self.similarities = self.similarities * similarities_weight

        self.similarities = self.similarities.to(self.teacher_model.device)
        self.teacher_model = None  # deallocate object to free up memory
        print()

    @staticmethod
    def _compute_embeddings(dataset: Iterable[str],
                            dataset_length: int,
                            tokenizer: PreTrainedTokenizer,
                            model: PreTrainedModel,
                            infer_batch_size: int,
                            dtype: torch.dtype = torch.float32,
                            init_eps: float = 1e-15) -> torch.FloatTensor:
        # embeddings collect running average of embeddings of output tokens
        # embeddings = init_eps * torch.rand((model.config.vocab_size, model.config.hidden_size), dtype=dtype)
        embeddings = torch.zeros((model.config.vocab_size, model.config.hidden_size), dtype=dtype, device=model.device)
        labels_macro_counts = torch.zeros(model.config.vocab_size, dtype=torch.long, device=model.device)

        for _ in tqdm(range(0, dataset_length, infer_batch_size), total=dataset_length//infer_batch_size,
                      desc="Generating embeddings"):
            texts_batch = list(itertools.islice(dataset, infer_batch_size))
            batch_embeddings = torch.zeros_like(embeddings, dtype=dtype, device=model.device)
            if not texts_batch:
                print("Skipping empty batch")
                break
            inputs = tokenizer(list(texts_batch), return_tensors="pt", truncation=True,
                               padding=True, max_length=2048).to(model.device)
            labels = inputs.input_ids[..., 1:].flatten(end_dim=1).contiguous()  # labels are used as embeddings mapping
            # with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden_states = outputs.hidden_states[-1][:, :-1].flatten(end_dim=1).type(dtype)

            # running average:
            # TODO: if we have doubts about correctness, check this on a toy example
            # https://stackoverflow.com/questions/57386257/how-can-i-compute-the-mean-of-values-selected-from-a-vector-a-from-an-indexing-v
            batch_labels_idx, batch_label_counts = labels.unique(return_counts=True)

            # average embeddings in batch
            labels_embs_micro_avg = batch_embeddings.index_add(0, labels, last_hidden_states)

            labels_embs_micro_avg[batch_labels_idx] /= batch_label_counts.unsqueeze(1)
            labels_weight_in_macro_avg = batch_label_counts / (batch_label_counts + labels_macro_counts[batch_labels_idx])
            labels_weight_in_macro_avg = labels_weight_in_macro_avg.type(dtype)

            # labels_weight_in_macro_avg = labels_norm / labels_macro_counts
            old_embs_weighted = embeddings[batch_labels_idx] * (1 - labels_weight_in_macro_avg.unsqueeze(1))
            new_embs_weighted = labels_embs_micro_avg[batch_labels_idx] * labels_weight_in_macro_avg.unsqueeze(1)
            embeddings[batch_labels_idx] = old_embs_weighted + new_embs_weighted

            # incrementally sum the counts of aggregated labels
            labels_macro_counts.index_add_(0, batch_labels_idx, batch_label_counts)

            if len(texts_batch) < infer_batch_size:
                break
        print("Constructed index for %s unique tokens from %s tokens seen in training corpus."
              % ((labels_macro_counts != 0).sum().item(), labels_macro_counts.sum().item()))
        return embeddings  # noqa type

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
        # TODO: try replacing with BCEWithLogitsLoss()
        loss_fct = torch.nn.BCEWithLogitsLoss()
        # loss_fct = torch.nn.CrossEntropyLoss()

        logits_f = lm_logit_outputs[..., :-1, :].flatten(end_dim=1)
        labels_f = labels[..., 1:].flatten(end_dim=1)

        soft_labels = self.similarities[labels_f]
        soft_labels[labels_f] = 1

        # TODO: construct training labels from the similarity of embeddings of {all} X {current_label}
        #  -> simply slice similarity matrix on a current next token and use that as target distribution

        # if lm_logit_outputs.shape == labels.shape:
        # for non-discrete targets, torch loss will not ignore the ignore_index targets,
        # so we exclude them manually
        ignored_idx = (labels_f == ignore_index)
        logits_f = logits_f[~ignored_idx]
        soft_labels = soft_labels[~ignored_idx]

        lm_loss = loss_fct(logits_f, soft_labels)
        # TODO: try with normalization:
        # lm_loss = loss_fct(logits_f.type(self.dtype), F.normalize(soft_labels))

        return lm_loss
