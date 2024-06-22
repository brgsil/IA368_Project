import gym
from gym import spaces
from gym.wrappers import GrayScaleObservation, FrameStack, ResizeObservation

import tella
from tella.curriculum import InterleavedEvalCurriculum


class RenderObservationWrapper(gym.ObservationWrapper):
    def __init__(
        self,
        env: gym.Env,
    ):
        gym.ObservationWrapper.__init__(self, env)

        self.render_history = []

        self.env.reset()
        pixels = self._render()
        pixels_spaces = spaces.Box(
            shape=pixels.shape, low=0, high=255, dtype=pixels.dtype
        )
        self.observation_space = pixels_spaces

    def observation(self, observation):
        pixel_observation = self._add_pixel_observation(observation)
        return pixel_observation

    def _add_pixel_observation(self, wrapped_observation):
        return self._render()

    def _render(self, *args, **kwargs):
        render = self.env.render(mode="rgb_array")
        if isinstance(render, list):
            self.render_history += render
        return render


class LunaLenderEnv(gym.Wrapper):
    def __init__(self, params={}):
        super().__init__(gym.make("LunarLander-v2", **params))
        # self.env = RenderObservationWrapper(self.env)
        # self.env = GrayScaleObservation(self.env)
        # self.env = ResizeObservation(self.env, 84)
        # self.env = FrameStack(self.env, 4)


class LunarLanderConstructor:
    def __init__(self, params={}):
        self.params = params

    def __call__(self):
        return LunaLenderEnv(params=self.params)


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
            "gravity": -2.0,
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
            "enable_wind": False,
            "wind_power": 20.0,
            "turbulence_power": 0.2,
        }
    ),
]

# test_env = LUNAR_ENVS[0]()
# print(test_env.observation_space)
# print(test_env.reset())
# print(test_env.step(0))
# print(test_env.step(0))


class LunarCurriculum(InterleavedEvalCurriculum):

    def eval_block(self):
        return tella.curriculum.simple_eval_block(
            [
                tella.curriculum.TaskVariant(
                    LUNAR_ENVS[i],
                    task_label=f"Lunar-Env-{i}",
                    variant_label=(
                        "Checkpoint" if i == len(LUNAR_ENVS) - 1 else "Default"
                    ),
                    num_episodes=5,
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
                    for j in range(19)
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
                    num_episodes=5,
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
            )
            for block in split_learn_block_per_task_variant(variants):
                yield block


def split_learn_block_per_task_variant(tasks):
    for variant in tasks:
        task_block = tella.curriculum.TaskBlock(variant.task_label, [variant])
        yield tella.curriculum.LearnBlock([task_block])
