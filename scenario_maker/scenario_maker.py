#!/usr/bin/env python3
import random
from dataclasses import dataclass
from typing import *

import numpy as np

import duckietown_world as dw
import geometry
from aido_schemas import protocol_scenario_maker, Scenario, ScenarioRobotSpec, RobotConfiguration, wrap_direct, Context
from duckietown_world import list_maps, load_map
from duckietown_world.world_duckietown.map_loading import _get_map_yaml
from duckietown_world.world_duckietown.sampling_poses import sample_good_starting_pose


@dataclass
class MyScenario(Scenario):
    scenario_name: str
    environment: str
    robots: Dict[str, ScenarioRobotSpec]


@dataclass
class MyConfig:
    maps: Tuple[str] = ('4way',)
    scenarios_per_map: int = 1
    robots_npcs: int = 0
    robots_pcs: int = 1
    theta_tol_deg: float = 30.0
    dist_tol_m: float = 0.3
    min_dist: float = 0.5
    only_straight: bool = True


@dataclass
class MyState:
    scenarios_to_go: List[MyScenario]


class SimScenarioMaker:
    config: MyConfig = MyConfig()
    state: MyState = MyState([])

    def init(self, context: Context):
        pass

    def _create_scenarios(self, context: Context):
        available = list_maps()
        for map_name in self.config.maps:
            if not map_name in available:
                msg = f'Cannot find map name "{map_name}, know {available}'
                raise Exception(msg)

            yaml_str: str = _get_map_yaml(map_name)

            po = load_map(map_name)

            for i in range(self.config.scenarios_per_map):
                scenario_name = f'{map_name}-{i}'

                nrobots = self.config.robots_npcs + self.config.robots_pcs
                poses = sample_many_good_starting_poses(po, nrobots,
                                                        only_straight=self.config.only_straight,
                                                        min_dist=self.config.min_dist)

                poses_pcs = poses[:self.config.robots_pcs]
                poses_npcs = poses[self.config.robots_pcs:]

                robots = {}
                for i in range(self.config.robots_pcs):
                    pose = poses_pcs[i]
                    vel = geometry.se2_from_linear_angular([0, 0], 0)

                    robot_name = 'ego' if i == 0 else "player%d" % i
                    configuration = RobotConfiguration(pose=pose, velocity=vel)

                    robots[robot_name] = ScenarioRobotSpec(description='Playable robot',
                                                            playable=True,
                                                            configuration=configuration)

                for i in range(self.config.robots_npcs):
                    pose = poses_npcs[i]
                    vel = geometry.se2_from_linear_angular([0, 0], 0)

                    robot_name = "npc%d" % i
                    configuration = RobotConfiguration(pose=pose, velocity=vel)

                    robots[robot_name] = ScenarioRobotSpec(description='NPC robot',
                                                           playable=False,
                                                           configuration=configuration)

                ms = MyScenario(scenario_name=scenario_name, environment=yaml_str, robots=robots)
                self.state.scenarios_to_go.append(ms)

    def on_received_seed(self, context: Context, data: int):
        context.info(f'seed({data})')
        np.random.seed(data)
        random.seed(data)

        self._create_scenarios(context)

    def on_received_next_scenario(self, context: Context):
        if self.state.scenarios_to_go:
            scenario = self.state.scenarios_to_go.pop(0)
            context.write('scenario', scenario)
        else:
            context.write('finished', None)

    def finish(self, context: Context):
        pass


def sample_many_good_starting_poses(po: dw.PlacedObject, nrobots: int, only_straight: bool, min_dist: float) -> List[
    np.ndarray]:
    poses = []

    def far_enough(pose):
        for p in poses:
            if distance_poses(p, pose) < min_dist:
                return False
        return True

    while len(poses) < nrobots:
        pose = sample_good_starting_pose(po, only_straight=only_straight)
        if far_enough(pose):
            poses.append(pose)
    return poses


def distance_poses(q1, q2):
    SE2 = geometry.SE2
    d = SE2.multiply(SE2.inverse(q1), q2)
    t, _a = geometry.translation_angle_from_SE2(d)
    return np.linalg.norm(t)


def main():
    node = SimScenarioMaker()
    protocol = protocol_scenario_maker
    protocol.outputs['scenario'] = MyScenario
    wrap_direct(node=node, protocol=protocol)


if __name__ == '__main__':
    main()
