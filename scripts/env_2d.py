import numpy as np
import cv2

OBS_SIZE = 4
N_TYPES = 3
INTERACTION_DISTANCE = 0.1

def sample_pos(spec):
    pos_min = np.array(spec['min'])
    pos_max = np.array(spec['max'])
    return (np.random.random(2) * (pos_max - pos_min)) + pos_min

def zero_pad(array, size):
    return np.pad(array, (0, size - len(array)), 'constant', constant_values=0)

def get_obs_token(obj):
    """
    Returns a pair (category_onehot, obs_token)
    """
    return (
        np.eye(N_TYPES, dtype=np.bool)[obj._typeid],
        zero_pad(obj.get_obs(), OBS_SIZE)
    )

def normalize_pos(canvas, min_bounds, max_bounds, pos):
    max_px = canvas.shape[:2]
    pos_relative = (pos - min_bounds) / (max_bounds - min_bounds)
    return np.array(pos_relative * max_px, dtype=int)

class Robot:
    _typeid = 0

    @staticmethod
    def from_json(data):
        return Robot(sample_pos(data['pos']))

    def __init__(self, pos, max_speed=0.05):
        self.pos = pos
        self.max_speed = max_speed
        self.inventory = None

    def update(self, act):
        delta = np.array(act[:2])
        move_speed = np.linalg.norm(delta)
        if move_speed > self.max_speed:
            delta = delta * (self.max_speed / move_speed)
        self.pos += delta

    def get_obs(self):
        if self.inventory is None:
            inventory = [0]
        else:
            inventory = self.inventory.get_local_obs()
        state_vec = np.array([
            *self.pos,
            *inventory
        ])
        return state_vec

    def render(self, canvas, min_bounds, max_bounds):
        cv2.circle(
            canvas,
            center=normalize_pos(canvas, min_bounds, max_bounds, self.pos),
            radius=3, color=(255, 0, 0), thickness=-1
        )

class Container:
    _typeid = 1

    @staticmethod
    def from_json(data, numeric_id):
        return Container(numeric_id, sample_pos(data['pos']), data.get('capacity', 1))

    def __init__(self, numeric_id, pos, capacity, items=None):
        self.id = numeric_id
        self.pos = pos
        self.capacity = capacity
        if items is None:
            self.items = []
        else:
            self.items = items

    def get_obs(self):
        return np.array([
            *self.pos,
            len(self.items),
            self.capacity
        ])

    def can_emplace(self, item):
        return len(self.items) < self.capacity

    def emplace(self, item):
        self.items.append(item)

    def can_pickup(self):
        return len(self.items) > 0

    def pickup(self):
        return self.items.pop(-1)

    def render(self, canvas, min_bounds, max_bounds):
        midpoint = normalize_pos(canvas, min_bounds, max_bounds, self.pos)
        p1 = midpoint - [4, 4]
        p2 = midpoint + [4, 4]
        cv2.rectangle(canvas, p1, p2, color=(0, 0, 0), thickness=2)

class Item:
    _typeid = 2

    @staticmethod
    def from_json(data, numeric_id):
        # By convention, type cannot be zero. (0 is no item)
        return Item(numeric_id, sample_pos(data['pos']), data.get('type', 1))

    def __init__(self, numeric_id, pos, item_type):
        self.id = numeric_id
        self.pos = pos
        self.type = item_type

    def get_local_obs(self):
        return [self.type]

    def get_obs(self):
        return self.pos

    def can_pickup(self):
        return True

    def pickup(self):
        return self

    def render(self, canvas, min_bounds, max_bounds):
        cv2.circle(
            canvas,
            center=normalize_pos(canvas, min_bounds, max_bounds, self.pos),
            radius=2, color=(0, 0, 255), thickness=-1
        )

class World2d:

    def __init__(self, data):
        self.data = data

    def reset(self):
        self.robot = Robot.from_json(self.data['robot'])
        self.containers = [
            Container.from_json(d, i) for i, d in enumerate(self.data['container'])
        ]
        self.items = [
            Item.from_json(d, i) for i, d in enumerate(self.data['item'])
        ]
        return self.get_obs()

    def get_closest_item(self, target_pos):
        closest_dist = np.inf
        closest_item = None
        item_index = None
        for item in self.containers:
            dist = np.linalg.norm(target_pos - item.pos)
            if dist < closest_dist:
                closest_item = item
                closest_dist = dist

        for i, item in enumerate(self.items):
            dist = np.linalg.norm(target_pos - item.pos)
            if dist < closest_dist:
                closest_item = item
                closest_dist = dist
                item_index = i
        return closest_dist, closest_item, item_index
    
    def update(self, act):
        # Do this twice; so that the robot can "act on the observation".
        closest_dist, closest_item, item_index = self.get_closest_item(self.robot.pos)

        if act[2] > 0.5:
            # Pickup item.
            if (
                self.robot.inventory is None and
                closest_dist < INTERACTION_DISTANCE and
                closest_item.can_pickup()
            ):
                # Container pickup() uses pop() to remove from list
                self.robot.inventory = closest_item.pickup()

                # A bit hacky: logic flow not unified between items and container pickup
                if item_index is not None:
                    self.items.pop(item_index)

        elif act[2] < -0.5:
            # Drop item.
            if self.robot.inventory is not None:
                # If you can place the item into a container....
                if (
                    closest_dist < INTERACTION_DISTANCE and
                    type(closest_item) == Container and
                    closest_item.can_emplace(self.robot.inventory)
                ):
                    closest_item.emplace(self.robot.inventory)
                    self.robot.inventory = None
                # Otherwise, just drop it on the ground
                else:
                    drop_item = self.robot.inventory
                    # This can delete information from the robot's sensors...
                    drop_item.pos = np.copy(self.robot.pos)
                    self.items.append(drop_item)
                    self.robot.inventory = None

        self.robot.update(act)
        res = self.get_obs()
        return res

    def get_obs(self):
        closest_dist, closest_item, item_index = self.get_closest_item(self.robot.pos)

        return {
            'robot': get_obs_token(self.robot),
            'containers': [
                get_obs_token(c) for c in self.containers
            ],
            'items': [
                get_obs_token(i) for i in self.items
            ],
            # NOTE: for now this will not work. The above tokens are
            # "ok" because they are all different types.
            #'closest': get_obs_token(closest_item)
            'closest': closest_item
        }


    def render(self, resolution=256):
        canvas = np.ones((resolution, resolution, 3), dtype=np.uint8) * 255
        min_bounds = np.array(self.data['bounds']['min'])
        max_bounds = np.array(self.data['bounds']['max'])
        for c in self.containers:
            c.render(canvas, min_bounds, max_bounds)
        self.robot.render(canvas, min_bounds, max_bounds)
        for i in self.items:
            i.render(canvas, min_bounds, max_bounds)
        return canvas


def tokenize_obs(obs, pad_to_size=None):
    tokens = []
    categories = []
    categories.append(obs['robot'][0])
    tokens.append(obs['robot'][1])
    for container_category, container_token in obs['containers']:
        categories.append(container_category)
        tokens.append(container_token)
    for item_category, item_token in obs['items']:
        categories.append(item_category)
        tokens.append(item_token)
    if pad_to_size is not None:
        while len(tokens) < pad_to_size:
            categories.append(np.zeros(N_TYPES, dtype=np.bool))
            tokens.append(np.zeros(OBS_SIZE))
    return (
        np.stack(tokens),
        np.stack(categories)
    )
