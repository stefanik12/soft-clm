import copy
import logging
from typing import Iterator, List, Sequence

from adaptor.objectives.objective_base import Objective
from adaptor.schedules import Schedule
from adaptor.utils import AdaptationArguments


logger = logging.getLogger()


class ContinuedSchedule(Schedule):

    label = "continued"

    def __init__(self,
                 objectives: List[Objective],
                 args: AdaptationArguments,
                 extra_eval_objectives: Sequence[Objective] = (),
                 rho_selection_strategy: str = "incremental"):
        assert len(objectives) == 1, "Expected just one Rho objective in this Schedule"
        assert len(extra_eval_objectives) == 1, ("In this Schedule, eval_objective is used "
                                                 "for early-stopping other objectives")
        self.rho_selection_strategy = rho_selection_strategy
        super().__init__(objectives, args)

    def _sample_objectives(self, split: str) -> Iterator[Objective]:
        """
        Sample objectives in a sequential order - each objective is sampled for its `dataset_length` steps.

        :param split: data split to iterate. `train` or `eval`. Currently, Schedule base uses only "train".
        :return: Iterator over the references to objectives.
        """
        # infinite loop - termination is determined by _should_stop() + _combine_datasets()
        train_objective = list(self.objectives["train"].values())[0]
        eval_objective = list(self.objectives["eval"].values())[0]

        train_objective.has_converged = eval_objective.has_converged
        # TODO: this will not work with distillation because the reference/teacher models
        #  for rho and distillation must be different
        generation = 0
        while True:
            # plan:
            # 1. assume one train and one eval objective
            # 2. fix the initial model from eval objective -- already fixed in eval_objective.teacher_model
            # 4. set the models for rho eval objective (init model; *independent* *copy* of train objective model)

            # TODO: is this ok?
            #  -> note: generation_start_model is only set as a reference model *after* the first generation of training
            generation_start_model = copy.deepcopy(train_objective.compatible_head_model)

            # actual training happens in this for-loop
            for _ in range(train_objective.dataset_length[split]):
                if train_objective in self.converged_objectives and not self.args.log_converged_objectives:
                    continue
                yield train_objective

            # 5. train until convergence of train objective on eval objective
            # -> should be done by linking train_objective's `has_converged` to eval_objective

            # 6. reset eval objective
            self.should_stop = False
            eval_objective.evaluations_history = []
            eval_objective.loss_history = []
            # 7. change train and eval models according to the strategy
            if self.rho_selection_strategy == "incremental":
                # incremental strategy moves reference models one generation ahead
                # -> Rho data selection will pick the data with the largest "learning" in the recent generation
                train_objective.teacher_model = generation_start_model
                # eval objective picks the data consistently with the training objective
                eval_objective.teacher_model = train_objective.teacher_model
                # the model must be shared between the training and evaluation objective
                eval_objective.compatible_head_model = train_objective.compatible_head_model

            # 8. continue training
            logger.warning("Finished training Rho generation %s", generation)
            generation += 1
