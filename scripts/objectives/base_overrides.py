from typing import Optional, Dict, Any

import torch
from adaptor.lang_module import LangModule
from adaptor.objectives.objective_base import Objective
from adaptor.objectives.seq2seq import Sequence2Sequence, SequentialMixin


class ExperimentOverrides(Objective):

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
