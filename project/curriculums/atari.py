import typing
import numpy as np
import tella


class SimpleAtariSequenceCurriculum(tella.curriculum.InterleavedEvalCurriculum):
    TASKS = ["RoadRunner", "Boxing", "Jamesbond"]

    def eval_block(self) -> tella.curriculum.EvalBlock:
        rng = np.random.default_rng(self.eval_rng_seed)
        return tella.curriculum.simple_eval_block(
            [
                tella.curriculum.TaskVariant(
                    tella._curriculums.atari.environments.ATARI_TASKS[task_label],
                    task_label=task_label,
                    num_episodes=2,
                    rng_seed=rng.bit_generator.random_raw(),
                )
                for task_label in self.TASKS
            ]
        )

    def learn_blocks(self) -> typing.Iterable[tella.curriculum.LearnBlock]:
        for task_label in self.TASKS:
            yield tella.curriculum.simple_learn_block(
                [
                    tella.curriculum.TaskVariant(
                        tella._curriculums.atari.environments.ATARI_TASKS[task_label],
                        task_label=task_label,
                        variant_label=f"Iter_{i}",
                        num_steps=5,
                        rng_seed=self.rng.bit_generator.random_raw(),
                    )
                    for i in range(2)
                ]
            )
