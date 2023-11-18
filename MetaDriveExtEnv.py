from metadrive import MetaDriveEnv
from typing import Union, Optional
import numpy as np
from metadrive.utils import Config


class MetaDriveInfoEnv(MetaDriveEnv):
    def __init__(self, config: Union[dict, None] = None):
        self.default_config_copy = Config(self.default_config(), unchangeable=True)
        super(MetaDriveInfoEnv, self).__init__(config)

    def step(self, *args, **kwargs):
        # single agent
        ret = super(MetaDriveInfoEnv, self).step(*args, **kwargs)
        observation, reward, terminated, truncated, step_infos = ret
        step_infos['observation'] = observation
        step_infos['done'] = terminated or truncated
        return observation, reward, terminated, truncated, step_infos

    def reset(self, seed: Union[None, int] = None, options: Optional[dict] = None):
        ret = super(MetaDriveInfoEnv, self).reset(seed, options)
        observation, step_infos = ret
        step_infos['observation'] = observation
        return observation, step_infos


class MetaDriveTrajEnv(MetaDriveEnv):
    pass


if __name__ == '__main__':

    def _act(env, action):
        assert env.action_space.contains(action)
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs)
        assert np.isscalar(reward)
        assert isinstance(info, dict)


    env = MetaDriveInfoEnv()
    try:
        obs, reset_info = env.reset()
        assert env.observation_space.contains(obs)
        _act(env, env.action_space.sample())
        for x in [-1, 0, 1]:
            env.reset()
            for y in [-1, 0, 1]:
                _act(env, [x, y])
    finally:
        env.close()
