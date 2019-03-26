#!/usr/bin/env python3
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from aido_schemas import EpisodeStart, protocol_agent_duckiebot1, PWMCommands, Duckiebot1Commands, LEDSCommands, RGB, \
    wrap_direct, Context, Duckiebot1Observations, JPGImage


@dataclass
class ImitationAgentConfig:
    pwm_left_interval: Tuple[int, int] = (0.2, 0.3)
    pwm_right_interval: Tuple[int, int] = (0.2, 0.3)


class ImitationAgent:
    config: ImitationAgentConfig = ImitationAgentConfig()

    def init(self, context: Context):
        context.info('init()')

    def on_received_seed(self, data: int):
        np.random.seed(data)

    def on_received_episode_start(self, context: Context, data: EpisodeStart):
        context.info(f'Starting episode "{data.episode_name}".')

    def on_received_observations(self, data: Duckiebot1Observations):
        camera: JPGImage = data.camera
        _rgb = jpg2rgb(camera.jpg_data)

    def on_received_get_commands(self, context: Context):
        l, u = self.config.pwm_left_interval
        pwm_left = np.random.uniform(l, u)
        l, u = self.config.pwm_right_interval
        pwm_right = np.random.uniform(l, u)

        grey = RGB(0.0, 0.0, 0.0)
        led_commands = LEDSCommands(grey, grey, grey, grey, grey)
        pwm_commands = PWMCommands(motor_left=pwm_left, motor_right=pwm_right)
        commands = Duckiebot1Commands(pwm_commands, led_commands)
        context.write('commands', commands)

    def finish(self, context: Context):
        context.info('finish()')


def jpg2rgb(image_data: bytes) -> np.ndarray:
    """ Reads JPG bytes as RGB"""
    from PIL import Image
    import io
    im = Image.open(io.BytesIO(image_data))
    im = im.convert('RGB')
    data = np.array(im)
    assert data.ndim == 3
    assert data.dtype == np.uint8
    return data


def main():
    node = ImitationAgent()
    protocol = protocol_agent_duckiebot1
    wrap_direct(node=node, protocol=protocol)


if __name__ == '__main__':
    main()
