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

lang_module = LangModule(model_path)
# lang_module = LangModule("facebook/nllb-200-distilled-600M")


# lang_module = LangModule("stas/mt5-tiny-random")

class BaselineCLM(CausalLanguageModeling, ExperimentOverrides):
    pass


class DistilledCLM(BaselineCLM, Distillation):
    pass


objective_kwargs = {
    "lang_module": lang_module,
    "batch_size": 1,
    "texts_or_path": "/Users/xstefan3/PycharmProjects/mammoth_lora/data/example_data_dir/eng-fur/test.src",
    "val_texts_or_path": "/Users/xstefan3/PycharmProjects/mammoth_lora/data/example_data_dir/eng-fur/test.src",
    "source_lang_id": "eng_Latn",
    "target_lang_id": "eng_Latn",
    # "teacher_model": transformers.AutoModelForCausalLM.from_pretrained(model_path)
}

train_objectives = [SoftCLM(**objective_kwargs)]

# Add pad token to all models if using pythia
if "pythia" in model_path:
    train_objectives[0].compatible_head_model.pad_token = "<|endoftext|>"
    train_objectives[0].tokenizer.pad_token = "<|endoftext|>"
if hasattr(train_objectives[0], "teacher_model") and "pythia" in train_objectives[0].teacher_model.name_or_path:
    train_objectives[0].teacher_model.pad_token = "<|endoftext|>"


training_arguments = AdaptationArguments(output_dir="adaptation_output_dir",
                                         stopping_strategy=StoppingStrategy.ALL_OBJECTIVES_CONVERGED,
                                         do_train=True,
                                         do_eval=True,
                                         gradient_accumulation_steps=1,
                                         logging_steps=1,
                                         num_train_epochs=2,
                                         no_cuda=True)
schedule = ParallelSchedule(train_objectives, training_arguments)

adapter = Adapter(lang_module, schedule, training_arguments)
adapter.train()

print("Done")
