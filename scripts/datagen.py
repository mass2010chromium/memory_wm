import json

import mediapy
import numpy as np
import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from env_2d import (
    INTERACTION_DISTANCE, OBS_SIZE, N_TYPES,
    sample_pos, World2d, tokenize_obs
)

INTERACTION_PROB = 0.5  # Per tick
DROP_PROB = 0.01  # Per tick
MAX_TOKENS = 5

features = {
    "observation.token_mask": {
        "dtype": "bool",
        "shape": (MAX_TOKENS,),
    },
    "observation.tokens": {
        "dtype": "float32",
        "shape": (MAX_TOKENS, OBS_SIZE),
    },
    "observation.token_categories": {
        "dtype": "bool",
        "shape": (MAX_TOKENS, N_TYPES),
    },
    "action": {
        "dtype": "float32",
        "shape": (3,),
    }
}

def gen_trajectory(world: World2d, dataset: LeRobotDataset):
    obs = world.reset()
    max_speed = world.robot.max_speed
    speed_multiple = (np.random.random() * 0.95) + 0.05
    chosen_speed = speed_multiple * max_speed

    target_point = sample_pos(world.data['bounds'])

    def add_frame(obs, action):
        obs_tokens, obs_categories = tokenize_obs(obs, pad_to_size=MAX_TOKENS)
        token_mask = np.zeros(MAX_TOKENS, dtype=np.bool)
        token_mask[0:1 + len(obs['containers']) + len(obs['items'])] = 1
        dataset.add_frame({
            "observation.token_mask": token_mask,
            "observation.tokens": obs_tokens.astype(np.float32),
            "observation.token_categories": obs_categories,
            "action": action.astype(np.float32),
            "task": "Do something useful"
        })

    has_obs = False
    while np.linalg.norm(world.robot.pos - target_point) > 1e-5:
        motion = target_point - world.robot.pos
        distance = np.linalg.norm(motion)
        motion = motion * (min(chosen_speed, distance) / distance)
        closest_obj = obs['closest']

        action = np.zeros(3)
        action[0:2] = motion
        if np.linalg.norm(closest_obj.pos - world.robot.pos) < INTERACTION_DISTANCE:
            if np.random.random() < INTERACTION_PROB * speed_multiple:
                if world.robot.inventory is None:
                    action[2] = 1.0
                else:
                    action[2] = -1.0
        if world.robot.inventory is not None:
            if np.random.random() < DROP_PROB * speed_multiple:
                action[2] = -1.0
        add_frame(obs, action)
        obs = world.update(action)
        has_obs = True
    if has_obs:
        add_frame(obs, np.zeros(3))
        dataset.save_episode()
    return has_obs


if __name__ == "__main__":
    with open("world.json", "r") as datafile:
        data = json.load(datafile)
    world = World2d(data)
    world.reset()
    img = world.render()
    mediapy.write_image('out_image.png', img)

    dataset = LeRobotDataset.create(
        repo_id="local/world2d",
        fps=30,
        features=features,
        root="./world2d",
        robot_type="custom",
        use_videos=True
    )
    for i in tqdm.trange(10000):
        while not gen_trajectory(world, dataset):
            print("Started on the goal, retry")
    dataset.finalize()
    print(dataset[0])
    print("----------")

    del dataset
    dataset = LeRobotDataset("local/world2d", root="./world2d")
    print(dataset[0])

