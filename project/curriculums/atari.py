import typing
import numpy as np
import tella


class SimpleAtariSequenceCurriculum(tella.curriculum.InterleavedEvalCurriculum):
    #TASKS = ["RoadRunner", "Boxing", "Jamesbond"]
    TASKS = ["RoadRunner", "Jamesbond"]
    #TASKS = [ "Boxing", "Jamesbond"]

    def eval_block(self) -> tella.curriculum.EvalBlock:
        rng = np.random.default_rng(self.eval_rng_seed)
        return tella.curriculum.simple_eval_block(
            [
                tella.curriculum.TaskVariant(
                    tella._curriculums.atari.environments.ATARI_TASKS[task_label],
                    task_label=task_label,
                    variant_label=(
                        "Checkpoint" if task_label == self.TASKS[-1] else "Default"
                    ),
                    num_episodes=30,
                    rng_seed=rng.bit_generator.random_raw(),
                )
                for task_label in self.TASKS
            ]
        )

    def learn_blocks(self) -> typing.Iterable[tella.curriculum.LearnBlock]:
        for variants in construct_variants(self.TASKS, self.rng):
            for block in split_learn_block_per_task_variant(variants):
                yield block

class SimpleAtariSequenceCurriculumPPO(tella.curriculum.InterleavedEvalCurriculum):
    TASKS = ["RoadRunner", "Boxing", "Jamesbond"]

    def eval_block(self) -> tella.curriculum.EvalBlock:
        rng = np.random.default_rng(self.eval_rng_seed)
        return tella.curriculum.simple_eval_block(
            [
                tella.curriculum.TaskVariant(
                    tella._curriculums.atari.environments.ATARI_TASKS[task_label],
                    task_label=task_label,
                    variant_label=(
                        "Checkpoint" if task_label == self.TASKS[-1] else "Default"
                    ),
                    num_episodes=30,
                    rng_seed=rng.bit_generator.random_raw(),
                )
                for task_label in self.TASKS
            ]
        )

    def learn_blocks(self) -> typing.Iterable[tella.curriculum.LearnBlock]:
        for task_label in self.TASKS:
            variants = [
                        tella.curriculum.TaskVariant(
                            tella._curriculums.atari.environments.ATARI_TASKS[task_label],
                            task_label=task_label,
                            variant_label="PPO_Train",
                            num_steps=400_000,
                            rng_seed=self.rng.bit_generator.random_raw(),
                        )
                        for _ in range(20)
                    ]
            for block in split_learn_block_per_task_variant(variants):
                yield block

def split_learn_block_per_task_variant(tasks):
    for variant in tasks:
        task_block = tella.curriculum.TaskBlock(variant.task_label, [variant])
        yield tella.curriculum.LearnBlock([task_block])


def construct_variants(tasks_labels, rng):
    yield (
        [
            tella.curriculum.TaskVariant(
                tella._curriculums.atari.environments.ATARI_TASKS[tasks_labels[0]],
                task_label=tasks_labels[0],
                variant_label="STM_Train",
                num_steps=600_000,
                rng_seed=rng.bit_generator.random_raw(),
            )
            for _ in range(20)
        ]
        + [
            tella.curriculum.TaskVariant(
                tella._curriculums.atari.environments.ATARI_TASKS[tasks_labels[0]],
                task_label=tasks_labels[0],
                variant_label="LTM_Last",
                num_steps=750_000,
                rng_seed=rng.bit_generator.random_raw(),
            )
        ]
    )

    for task_label in tasks_labels[1:]:
        yield (
            [
                tella.curriculum.TaskVariant(
                    tella._curriculums.atari.environments.ATARI_TASKS[tasks_labels[0]],
                    task_label=tasks_labels[0],
                    variant_label="STM_Train",
                    num_steps=600_000,
                    rng_seed=rng.bit_generator.random_raw(),
                )
                for _ in range(20)
            ]
            + [
                tella.curriculum.TaskVariant(
                    tella._curriculums.atari.environments.ATARI_TASKS[task_label],
                    task_label=task_label,
                    variant_label=f"LTM_Iter_{i}",
                    num_steps=37_500,
                    rng_seed=rng.bit_generator.random_raw(),
                )
                for i in range(19)
            ]
            + [
                tella.curriculum.TaskVariant(
                    tella._curriculums.atari.environments.ATARI_TASKS[task_label],
                    task_label=task_label,
                    variant_label="LTM_Last",
                    num_steps=37_500,
                    rng_seed=rng.bit_generator.random_raw(),
                )
            ]
        )
