import copy
import logging
from typing import Iterator, List, Sequence, Iterable, Dict, Any

from adaptor.objectives.objective_base import Objective
from adaptor.schedules import Schedule, SequentialSchedule, ParallelSchedule
from adaptor.utils import AdaptationArguments


logger = logging.getLogger()


class ContinuedSchedule(ParallelSchedule):

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

    def _combine_datasets(self, split: str) -> Iterable[Dict[str, Any]]:
        """
        Constructs combined iterator over the datasets of all objectives,
        according to the implemented `_sample_objectives`.
        This main training iteration is upper-bound by a `num_epochs` over a full data set.
        :param split: data split to iterate.
        :return: Iterator over samples of selected split.
        """

        # plan:
        # 1. assume one train and one eval objective
        # 2. fix the initial model from eval objective -- already fixed in eval_objective.teacher_model
        # 4. set the models for rho eval objective
        train_objective = list(self.objectives["train"].values())[0]
        eval_objective = list(self.objectives["eval"].values())[0]
        train_objective.has_converged = eval_objective.has_converged

        # TODO: is this ok?
        #  -> note: generation_start_model is only set as a reference model *after* the first generation of training
        generation_start_model = copy.deepcopy(train_objective.compatible_head_model)
        generation = 0
        print("Starting training Rho generation %s", generation)

        if split == "train":
            objective_sampler = self._sample_objectives(split)
        else:
            # evaluation split uses simple, sequential evaluation over objectives
            objective_sampler = SequentialSchedule.single_iteration_eval_sampling(self.objectives["eval"].values())

        objectives_data_samplers = {obj: self._sample_objective_dataset(obj, obj_i, split)
                                    for obj_i, obj in enumerate(self.objectives[split].values())}
        for i, objective in enumerate(objective_sampler):
            try:
                yield next(objectives_data_samplers[objective])
            except StopIteration:
                # TODO: evaluation routine was reported to have raised StopIteration, we should find out why
                # logger.warning("Scheduler %s + Objective %s raised StopIteration.", self, objective)
                continue

            # 5. train until convergence of train objective on eval objective
            # -> should be done by linking train_objective's `has_converged` to eval_objective
            if self.should_stop:

                # 6. reset eval objective
                eval_objective.evaluations_history = {"train": {}, "eval": {}}
                eval_objective.loss_history = {"train": [], "eval": []}
                # 7. change train and eval models according to the strategy
                if self.rho_selection_strategy == "incremental":
                    new_generation_start_model = copy.deepcopy(train_objective.compatible_head_model)
                    # incremental strategy moves reference models one generation ahead
                    # -> Rho data selection will pick the data with the largest "learning" in the recent generation
                    train_objective.teacher_model = generation_start_model
                    # eval objective picks the data consistently with the training objective
                    eval_objective.teacher_model = train_objective.teacher_model
                    # the model must be shared between the training and evaluation objective
                    eval_objective.compatible_head_model = train_objective.compatible_head_model

                # 8. continue training
                print("Finished training Rho generation %s" % generation)
                generation += 1

                self.should_stop = False
