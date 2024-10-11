# TODO: tomorrow:
#  sort out inheritance with new collator to soft_clm
#  merge distributed training
#  make it running on lumi

import argparse

import torch
import transformers
from adaptor.adapter import Adapter
from adaptor.lang_module import LangModule
from adaptor.schedules import ParallelSchedule
from adaptor.utils import StoppingStrategy, AdaptationArguments

from evaluators import LMHarnessEvaluator
from objectives.base_overrides import DistilledCLM, BaselineCLM
from objectives.soft_clm import SoftCLM

torch.multiprocessing.set_start_method('spawn')

parser = argparse.ArgumentParser()
parser.add_argument("--objective", help="Objective of training. One of `soft-clm`, `distilled-clm`, `clm`",
                    required=True, type=str)
parser.add_argument("--base_model", help="A model to initialize the training with", required=True, type=str)
parser.add_argument("--teacher_model", help="Teacher model", type=str, default="")
parser.add_argument("--batch_size", help="Used batch size", type=int, default=4)
parser.add_argument("--batch_aggregation", help="Batch aggregation", type=int, default=2)
parser.add_argument("--eval_steps", help="Eval steps", type=int, default=100)
args = parser.parse_args()

# model_path = "EleutherAI/pythia-160m"
# model_path = "EleutherAI/pythia-14m"

lang_module = LangModule(args.base_model)
# lang_module = LangModule("facebook/nllb-200-distilled-600M")


# lang_module = LangModule("stas/mt5-tiny-random")


TrainingObj = SoftCLM if args.objective == "soft-clm" else DistilledCLM if args.objective == "distilled-clm" else BaselineCLM

device = "cuda" if torch.cuda.is_available() else "cpu"

objective_kwargs = {
    "lang_module": lang_module,
    "batch_size": args.batch_size,
    "texts_or_path": "data/train_10M/all_shuf_1k_nonempty.train"
                     if device == "cpu" else "data/train_10M/all_shuf_nonempty.train",
    "val_texts_or_path": "data/dev/all_shuf_1k_nonempty.dev",
    "source_lang_id": "eng_Latn",
    "target_lang_id": "eng_Latn",
    "teacher_model": transformers.AutoModelForCausalLM.from_pretrained(args.teacher_model).to(device)
}
if args.objective == "clm":
    del objective_kwargs["teacher_model"]
    extra_eval_objectives = []
else:
    extra_eval_objectives = [BaselineCLM(**{k: v for k, v in objective_kwargs.items() if k != "teacher_model"})]

evaluators = LMHarnessEvaluator(tasks=['blimp_filtered'])

train_objectives = [TrainingObj(**objective_kwargs, val_evaluators=[evaluators])]

# Add pad token to all models if using pythia
if train_objectives[0].tokenizer.pad_token is None and train_objectives[0].tokenizer.pad_token_id is None:
    train_objectives[0].compatible_head_model.pad_token = "<|endoftext|>"
    train_objectives[0].tokenizer.pad_token = "<|endoftext|>"
# if hasattr(train_objectives[0], "teacher_model") and "pythia" in train_objectives[0].teacher_model.name_or_path:
#     train_objectives[0].teacher_model.pad_token = "<|endoftext|>"


training_arguments = AdaptationArguments(output_dir="adaptation_output_dir",
                                         stopping_strategy=StoppingStrategy.ALL_OBJECTIVES_CONVERGED,
                                         do_train=True,
                                         do_eval=True,
                                         gradient_accumulation_steps=args.batch_aggregation,
                                         eval_steps=args.eval_steps,
                                         evaluation_strategy="steps",
                                         logging_steps=1000,
                                         num_train_epochs=5,
                                         warmup_steps=1000,
                                         learning_rate=4e-4,
                                         save_steps=10000,
                                         bf16=False if device == "cpu" else False,
                                         no_cuda=True if device == "cpu" else False,
                                         )
schedule = ParallelSchedule(train_objectives, training_arguments, extra_eval_objectives)

adapter = Adapter(lang_module, schedule, training_arguments)

lang_module.reinitialize()

adapter.train()

print("Done")
