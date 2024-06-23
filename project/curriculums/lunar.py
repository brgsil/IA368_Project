import gym
from gym.wrappers.pixel_observation import PixelObservationWrapper
from copy import copy


import tella
from tella.curriculum import InterleavedEvalCurriculum


class LunarLanderConstructor:
    def __init__(self, params={}):
        self.params = params

    def __call__(self):
        return gym.make("LunarLander-v2", **self.params)


LUNAR_ENVS = [
    LunarLanderConstructor(
        params={
            "gravity": -10.0,
            "enable_wind": False,
            "wind_power": 15.0,
            "turbulence_power": 1.5,
        }
    ),
    LunarLanderConstructor(
        params={
            "gravity": -10.0,
            "enable_wind": True,
            "wind_power": 15.0,
            "turbulence_power": 1.5,
        }
    ),
    LunarLanderConstructor(
        params={
            "gravity": -6.0,
            "enable_wind": True,
            "wind_power": 5.0,
            "turbulence_power": 1.5,
        }
    ),
    LunarLanderConstructor(
        params={
            "gravity": -4.0,
            "enable_wind": False,
            "wind_power": 5.0,
            "turbulence_power": 1.5,
        }
    ),
    LunarLanderConstructor(
        params={
            "gravity": -11.99,
            "enable_wind": False,
            "wind_power": 5.0,
            "turbulence_power": 1.5,
        }
    ),
    LunarLanderConstructor(
        params={
            "gravity": -10.0,
            "enable_wind": True,
            "wind_power": 20.0,
            "turbulence_power": 0.2,
        }
    ),
    LunarLanderConstructor(
        params={
            "gravity": -9.0,
            "enable_wind": True,
            "wind_power": 2.0,
            "turbulence_power": 2.0,
        }
    ),
    LunarLanderConstructor(
        params={
            "gravity": -6.0,
            "enable_wind": True,
            "wind_power": 20.0,
            "turbulence_power": 0.2,
        }
    ),
    LunarLanderConstructor(
        params={
            "gravity": -11.0,
            "enable_wind": True,
            "wind_power": 10.0,
            "turbulence_power": 0.8,
        }
    ),
    LunarLanderConstructor(
        params={
            "gravity": -5.0,
            "enable_wind": True,
            "wind_power": 3.0,
            "turbulence_power": 2.0,
        }
    ),
]


class LunarCurriculum(InterleavedEvalCurriculum):

    def eval_block(self):
        variants = []
        return tella.curriculum.simple_eval_block(
                [
                    tella.curriculum.TaskVariant(
                        LUNAR_ENVS[i],
                        task_label=f"Lunar-Env-{i}",
                        variant_label=(
                            "Checkpoint" if i == len(LUNAR_ENVS) - 1 else "Default"
                        ),
                        num_episodes=20,
                        rng_seed=1234,
                    )
                    for i in range(len(LUNAR_ENVS))
                ]
            )

    def learn_blocks(self):
        for i in range(len(LUNAR_ENVS)):
            variants = (
                [
                    tella.curriculum.TaskVariant(
                        LUNAR_ENVS[i],
                        task_label=f"Lunar-Env-{i}",
                        variant_label="PPO_Train_STM",
                        num_steps=20_000,
                        rng_seed=1234,
                    )
                    for _ in range(50)
                ]
                + [
                    tella.curriculum.TaskVariant(
                        LUNAR_ENVS[i],
                        task_label=f"Lunar-Env-{i}",
                        variant_label=f"PPO_LTM_{j}",
                        num_steps=1_000,
                        rng_seed=1234,
                    )
                    for j in range(49)
                ]
                + [
                    tella.curriculum.TaskVariant(
                        LUNAR_ENVS[i],
                        task_label=f"Lunar-Env-{i}",
                        variant_label="PPO_LTM_Last",
                        num_steps=1_000,
                        rng_seed=1234,
                    )
                ]
            )
            for block in split_learn_block_per_task_variant(variants):
                yield block


class LunarCurriculumPPO(InterleavedEvalCurriculum):

    def eval_block(self):
        return tella.curriculum.simple_eval_block(
            [
                tella.curriculum.TaskVariant(
                    LUNAR_ENVS[i],
                    task_label=f"Lunar-Env-{i}",
                    variant_label=(
                        "Checkpoint" if i == len(LUNAR_ENVS) - 1 else "Default"
                    ),
                    num_episodes=20,
                    rng_seed=1234,
                )
                for i in range(len(LUNAR_ENVS))
            ]
        )

    def learn_blocks(self):
        for i in range(len(LUNAR_ENVS)):
            variants = [
                tella.curriculum.TaskVariant(
                    LUNAR_ENVS[i],
                    task_label=f"Lunar-Env-{i}",
                    variant_label="PPO_Train_STM",
                    num_steps=20_000,
                    rng_seed=1234,
                )
                for _ in range(50)
            ]
            for block in split_learn_block_per_task_variant(variants):
                yield block


def split_learn_block_per_task_variant(tasks):
    for variant in tasks:
        task_block = tella.curriculum.TaskBlock(variant.task_label, [variant])
        yield tella.curriculum.LearnBlock([task_block])
