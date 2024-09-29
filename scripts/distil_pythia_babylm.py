import torch
import transformers
from adaptor.adapter import Adapter
from adaptor.lang_module import LangModule
from adaptor.objectives.CLM import CausalLanguageModeling
from adaptor.objectives.distillation import Distillation
from adaptor.schedules import ParallelSchedule
from adaptor.utils import StoppingStrategy, AdaptationArguments

from objectives.base_overrides import ExperimentOverrides
from objectives.soft_clm import SoftCLM

model_path = "EleutherAI/pythia-160m"
# model_path = "EleutherAI/pythia-14m"

lang_module = LangModule(model_path)
# lang_module = LangModule("facebook/nllb-200-distilled-600M")


# lang_module = LangModule("stas/mt5-tiny-random")

class BaselineCLM(CausalLanguageModeling, ExperimentOverrides):
    pass


class DistilledCLM(BaselineCLM, Distillation):
    pass


TrainingObj = SoftCLM

device = "cuda" if torch.cuda.is_available() else "cpu"

objective_kwargs = {
    "lang_module": lang_module,
    "batch_size": 4,
    "texts_or_path": "data/train_10M/all_shuf_1k_nonempty.train",
    "val_texts_or_path": "data/dev/all_shuf_1k_nonempty.dev",
    "source_lang_id": "eng_Latn",
    "target_lang_id": "eng_Latn",
    "teacher_model": transformers.AutoModelForCausalLM.from_pretrained(model_path).to(device)
}
if TrainingObj == BaselineCLM:
    del objective_kwargs["teacher_model"]

train_objectives = [TrainingObj(**objective_kwargs)]

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
                                         gradient_accumulation_steps=1,
                                         eval_steps=10000,
                                         evaluation_strategy="steps",
                                         logging_steps=10,
                                         num_train_epochs=2,
                                         no_cuda=True if device == "cpu" else False,
                                         )
schedule = ParallelSchedule(train_objectives, training_arguments)

adapter = Adapter(lang_module, schedule, training_arguments)
adapter.train()

print("Done")
