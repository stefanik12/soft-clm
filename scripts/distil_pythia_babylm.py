# TODO: tomorrow:
#  sort out inheritance with new collator to soft_clm
#  merge distributed training
#  make it running on lumi

import argparse
import os

import torch
import transformers
from adaptor.adapter import Adapter
from adaptor.lang_module import LangModule
from adaptor.schedules import ParallelSchedule
from adaptor.utils import StoppingStrategy, AdaptationArguments

from evaluators import LMHarnessEvaluator
from objectives.base_overrides import DistilledCLM, BaselineCLM
from objectives.soft_clm import SoftCLM

parser = argparse.ArgumentParser()
parser.add_argument("--objective",
                    help="Objective of training. One of `soft-clm`, `distilled-clm`, `clm`, `clm+soft-clm`",
                    required=True, type=str)
parser.add_argument("--base_model", help="A model to initialize the training with", required=True, type=str)
parser.add_argument("--output_dir_root", help="A root dir for training checkpoints", required=True, type=str)
parser.add_argument("--teacher_model", help="Teacher model", type=str, default="")
parser.add_argument("--batch_size", help="Used batch size", type=int, default=4)
parser.add_argument("--lr", help="Used batch size", type=int, default=3e-4)
parser.add_argument("--batch_aggregation", help="Batch aggregation", type=int, default=2)
parser.add_argument("--eval_steps", help="Eval steps", type=int, default=100)
parser.add_argument("--train_texts", help="Training texts", type=str)
parser.add_argument("--val_texts", help="Perplexity validation texts", type=str)
parser.add_argument("--soft_clm_sim_weight", help="Similarities weight", type=float, default=1.0)

# TODO: this would be nice, but requires nontrivial rewrite of lm_eval.evaluator.simple_evaluate:
# parser.add_argument("--eval_tasks_root", help="Root directory of lm_eval's evaluation tasks", type=str)

args = parser.parse_args()

# model_path = "EleutherAI/pythia-160m"
# model_path = "EleutherAI/pythia-14m"

# debugging probabilities:
# inputs = tokenizer("Cheryl isn't insulting", return_tensors="pt")  # -> herself
# outputs = model(**inputs)
# sorted(zip(outputs.logits[0, -1].softmax(dim=-1), tokenizer.vocab), reverse=True)

lang_module = LangModule(args.base_model)

TrainingObj = SoftCLM if args.objective == "soft-clm" else DistilledCLM if args.objective == "distilled-clm" else BaselineCLM

objective_kwargs = {
    "lang_module": lang_module,
    "batch_size": args.batch_size,
    "texts_or_path": args.train_texts,
    "val_texts_or_path": args.val_texts,
    "source_lang_id": "eng_Latn",
    "target_lang_id": "eng_Latn",
    "teacher_model": transformers.AutoModelForCausalLM.from_pretrained(args.teacher_model)
}

if torch.cuda.is_available():
    objective_kwargs["teacher_model"] = objective_kwargs["teacher_model"].to("cuda:%s" % os.environ.get("LOCAL_RANK", 0))

evaluators = LMHarnessEvaluator(tasks=['blimp_filtered', 'ewok_filtered'], batch_size=args.batch_size,
                                debug_mode=args.base_model == "EleutherAI/pythia-14m")

train_objectives = []
extra_eval_objectives = []

if "soft-clm" in args.objective:
    train_objectives.append(SoftCLM(**objective_kwargs, val_evaluators=[evaluators], similarities_weight=args.soft_clm_sim_weight))
elif "distilled-clm" in args.objective:
    train_objectives.append(DistilledCLM(**objective_kwargs, val_evaluators=[evaluators]))

baseline_kwargs = {k: v for k, v in objective_kwargs.items() if k != "teacher_model"}
if not train_objectives:
    # no other training objectives -> avoid duplicate evaluations
    baseline_kwargs["val_evaluators"] = [evaluators]

baseline_clm_obj = BaselineCLM(**baseline_kwargs)

if args.objective in ["clm", "clm+soft-clm", "clm+distilled-clm"]:
    train_objectives.append(baseline_clm_obj)
else:
    extra_eval_objectives.append(baseline_clm_obj)


# Add pad token to all models if using pythia
if train_objectives[0].tokenizer.pad_token is None and train_objectives[0].tokenizer.pad_token_id is None:
    train_objectives[0].compatible_head_model.pad_token = "<|endoftext|>"
    train_objectives[0].tokenizer.pad_token = "<|endoftext|>"
# if hasattr(train_objectives[0], "teacher_model") and "pythia" in train_objectives[0].teacher_model.name_or_path:
#     train_objectives[0].teacher_model.pad_token = "<|endoftext|>"


training_arguments = AdaptationArguments(output_dir=os.path.join(args.output_dir_root,
                                                                 "%s_checkpoints" % "+".join([str(o) for o in train_objectives])),
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
                                         save_steps=5000,
                                         # note that on lumi, we overrode transformers.**.is_torch_bf16_gpu_available
                                         bf16=torch.cuda.is_available(),
                                         no_cuda=not torch.cuda.is_available(),
                                         )
schedule = ParallelSchedule(train_objectives, training_arguments, extra_eval_objectives)

adapter = Adapter(lang_module, schedule, training_arguments)

lang_module.reinitialize()

adapter.train()

print("Done")
