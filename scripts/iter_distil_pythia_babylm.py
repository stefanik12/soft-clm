# TODO: tomorrow:
#  sort out inheritance with new collator to soft_clm
#  merge distributed training
#  make it running on lumi

import argparse
import os

import torch
import transformers
import wandb
from adaptor.adapter import Adapter
from adaptor.lang_module import LangModule
from adaptor.schedules import ParallelSchedule
from adaptor.utils import StoppingStrategy, AdaptationArguments

from evaluators import LMHarnessEvaluator
from objectives.base_overrides import DistilledCLM
from objectives.continued_schedule import ContinuedSchedule

parser = argparse.ArgumentParser()
parser.add_argument("--base_model", help="A model to initialize the training with", required=True, type=str)
parser.add_argument("--output_dir_root", help="A root dir for training checkpoints", required=True, type=str)
parser.add_argument("--starting_checkpoint", help="Teacher model", type=str, default="")
parser.add_argument("--batch_size", help="Used batch size", type=int, default=4)
parser.add_argument("--lr", help="Used batch size", type=float, default=3e-4)
parser.add_argument("--batch_aggregation", help="Batch aggregation", type=int, default=2)
parser.add_argument("--eval_steps", help="Eval steps", type=int, default=100)
parser.add_argument("--eval_samples_per_task", help="Number of samples per eval task", type=int, default=None)
parser.add_argument("--train_texts", help="Training texts", type=str)
parser.add_argument("--val_texts", help="Perplexity validation texts", type=str)
parser.add_argument("--force_true_tokens", help="Set probs of true tokens in distillation to one.", type=str,
                    default="False")
parser.add_argument("--force_false_tokens", help="Set probs of false tokens in distillation to zero.", type=str,
                    default="False")
parser.add_argument("--restrict_loss_to_mask", help="Whether to use only non-masked tokend in training.", type=str,
                    default="False")
parser.add_argument("--norm_attention", help="Whether to replace attention with NormAttention", type=str,
                    default="False")
parser.add_argument("--rho_token_selection_ratio",
                    help="Proportion of the tokens with the largest loss difference between student and teacher, used in training.",
                    type=float, default=1.0)

# TODO: this would be nice, but requires nontrivial rewrite of lm_eval.evaluator.simple_evaluate:
# parser.add_argument("--eval_tasks_root", help="Root directory of lm_eval's evaluation tasks", type=str)

args = parser.parse_args()
args.force_true_tokens = args.force_true_tokens.lower() != "false"
args.force_false_tokens = args.force_false_tokens.lower() != "false"
args.norm_attention = args.norm_attention.lower() != "false"
args.restrict_loss_to_mask = args.restrict_loss_to_mask.lower() != "false"

wandb_logger = wandb.init(project="babylm", entity="transformersclub", group="distillation", config=args)

# model_path = "EleutherAI/pythia-160m"
# model_path = "EleutherAI/pythia-14m"

lang_module = LangModule(args.base_model)

TrainingObj = DistilledCLM

objective_kwargs = {
    "lang_module": lang_module,
    "batch_size": args.batch_size,
    "texts_or_path": args.train_texts,
    "val_texts_or_path": args.val_texts,
    "source_lang_id": "eng_Latn",
    "target_lang_id": "eng_Latn",
    "teacher_model": transformers.AutoModelForCausalLM.from_pretrained(args.starting_checkpoint),
    "restrict_loss_to_mask": args.restrict_loss_to_mask,
    "rho_token_selection_ratio": args.rho_token_selection_ratio
}
if torch.cuda.is_available():
    objective_kwargs["teacher_model"] = objective_kwargs["teacher_model"].to(
            "cuda:%s" % os.environ.get("LOCAL_RANK", 0))


evaluators = LMHarnessEvaluator(tasks=['blimp_filtered', 'ewok_filtered', 'super-glue-lm-eval-v1'],
                                batch_size=args.batch_size,
                                limit=args.eval_samples_per_task,
                                )

train_obj = DistilledCLM(val_evaluators=[evaluators],
                         force_true_tokens=args.force_true_tokens,
                         force_false_tokens=args.force_false_tokens,
                         **objective_kwargs
                         )

eval_obj = DistilledCLM(force_true_tokens=True,
                        force_false_tokens=True,
                        **objective_kwargs
                        )

train_objectives = [train_obj]
extra_eval_objectives = [eval_obj]

# Add pad token to all models if using pythia
if train_objectives[0].tokenizer.pad_token is None and train_objectives[0].tokenizer.pad_token_id is None:
    train_objectives[0].compatible_head_model.pad_token = "<|endoftext|>"
    train_objectives[0].tokenizer.pad_token = "<|endoftext|>"
# if hasattr(train_objectives[0], "teacher_model") and "pythia" in train_objectives[0].teacher_model.name_or_path:
#     train_objectives[0].teacher_model.pad_token = "<|endoftext|>"

if args.norm_attention:
    from modeling.norm_attention import norm_attention_from_existing_module
    from modeling.overrides import patch_modules_with_overrides

    orig_model = train_objectives[0].compatible_head_model
    orig_attn = next(obj for name, obj in orig_model.named_modules()
                     if name.endswith("attention") or name.endswith("attn"))

    new_attn_cls = norm_attention_from_existing_module(orig_attn.__class__)
    new_attn_model = patch_modules_with_overrides(orig_model, orig_attn.__class__, new_attn_cls,
                                                  {"min_scale": 3.})

    train_objectives[0].compatible_head_model = new_attn_model

checkpoints_subdir = "%s_checkpoints" % "+".join([str(o) for o in train_objectives]) \
    if not (args.force_true_tokens and args.force_true_tokens) else "forced_baseline_checkpoints"

training_arguments = AdaptationArguments(output_dir=os.path.join(args.output_dir_root, checkpoints_subdir),
                                         stopping_strategy=StoppingStrategy.ALL_OBJECTIVES_CONVERGED,
                                         do_train=True,
                                         do_eval=True,
                                         gradient_accumulation_steps=args.batch_aggregation,
                                         eval_steps=args.eval_steps,
                                         evaluation_strategy="steps",
                                         logging_steps=100,
                                         num_train_epochs=5,
                                         warmup_steps=1000,
                                         learning_rate=args.lr,
                                         save_steps=5000 if not torch.cuda.is_available() else args.eval_steps,
                                         save_total_limit=21,
                                         # note that on lumi, we overrode transformers.**.is_torch_bf16_gpu_available
                                         bf16=torch.cuda.is_available(),
                                         no_cuda=not torch.cuda.is_available(),
                                         max_steps=500 if not torch.cuda.is_available() else None,
                                         )
schedule = ContinuedSchedule(train_objectives, training_arguments, extra_eval_objectives)

adapter = Adapter(lang_module, schedule, training_arguments)

# lang_module.reinitialize()  # not used -- we assume a fixed, pre-initialized random model

adapter.train()  # TODO: training produces NaNs on local -> test that on remote

print("Done")
