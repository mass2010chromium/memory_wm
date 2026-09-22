"""
CEM planner feat. google gemini
"""

import os
SCRIPT_DIR = os.path.dirname(__file__)

import torch

class CEMPlanner:
    def __init__(self, action_dim, plan_horizon=12, num_candidates=1000, 
                 num_elites=100, num_iterations=5, clip_actions=None, device='cpu'):
        """
        Cross-Entropy Method (CEM) Planner for continuous control.
        """
        self.action_dim = action_dim
        self.plan_horizon = plan_horizon
        self.num_candidates = num_candidates
        self.num_elites = num_elites
        self.num_iterations = num_iterations
        self.clip_actions = clip_actions
        self.device = device

    def plan(self, initial_state, dynamics_model, reward_function):
        """
        Optimizes an action sequence using the CEM algorithm.
        
        Args:
            initial_state: Tensor of shape (state_dim,)
            dynamics_model: Callable f(state, action) -> next_state
            reward_function: Callable f(state, action) -> reward
            
        Returns:
            The first action of the optimized sequence.
        """
        # Initialize the mean and standard deviation of the action distribution
        mean = torch.zeros(self.plan_horizon, self.action_dim, device=self.device)
        std = torch.ones(self.plan_horizon, self.action_dim, device=self.device)

        for _ in range(self.num_iterations):
            # 1. Sample candidate action sequences from a Gaussian distribution
            # Shape: [num_candidates, plan_horizon, action_dim]
            noise = torch.randn(self.num_candidates, self.plan_horizon, self.action_dim, device=self.device)
            actions = mean + std * noise
            
            # Clip actions to the environment's allowable range
            if self.clip_actions:
                actions = self.clip_actions(actions)

            # 2. Evaluate candidates by simulating them through the models
            returns = torch.zeros(self.num_candidates, device=self.device)
            
            # Repeat the initial state for all candidates to evaluate them in parallel
            # Shape: [num_candidates, state_dim]
            current_states = initial_state.unsqueeze(0).repeat(self.num_candidates, 1)

            for t in range(self.plan_horizon):
                action_t = actions[:, t, :]
                
                # Compute rewards and transition to the next state
                rewards = reward_function(current_states, action_t)
                current_states = dynamics_model(current_states, action_t)
                
                # Accumulate the return (sum of rewards) for each candidate
                returns += rewards.squeeze()

            # 3. Select the "elite" candidates with the highest returns
            _, top_indices = torch.sort(returns, descending=True)
            elite_indices = top_indices[:self.num_elites]
            elite_actions = actions[elite_indices]

            # 4. Update the distribution parameters based on the elite candidates
            mean = elite_actions.mean(dim=0)
            # Add a small epsilon to the std to prevent premature convergence (collapse to 0)
            std = elite_actions.std(dim=0, unbiased=False) + 1e-5 

        # In receding horizon control, we only execute the first action of the best sequence
        return mean[0], mean


if __name__ == "__main__":
    import json
    import numpy as np

    import py_terminal_plotter as ptp

    from memory_wm.module import Predictor

    from env_2d import tokenize_obs, World2d, MAX_TOKENS

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    plotter = ptp.TerminalPlot(x_range=[0, 1], y_range=[0, 1])
    plotter.create_axes(title="Planing World")
    plotter.setup_image(256, 256, z_range=[0, 255])

    def load_model(model_config):
        out_dir = os.path.join(SCRIPT_DIR, "checkpoints")
        data = torch.load(os.path.join(out_dir, "299.pth"), weights_only=True)

        model = Predictor(**model_config).cuda()
        model.load_state_dict(data['model_state'])
        model.eval()
        return model, data['latent_cache']

    with open(os.path.join(SCRIPT_DIR, "world.json"), "r") as jf:
        data = json.load(jf)
    seed = 99
    np.random.seed(seed)
    world = World2d(data)
    last_obs = world.reset()
    max_speed = world.robot.max_speed

    with open(os.path.join(SCRIPT_DIR, "config", "model_config.json"), "r") as jf:
        config = json.load(jf)
    model, _ = load_model(config)

    def embed_obs(obs):
        obs_tokens, obs_categories, token_mask = tokenize_obs(obs, pad_to_size=MAX_TOKENS)
        with torch.no_grad():
            obs_embed = model.embed_obs(
                torch.tensor(obs_tokens).float().unsqueeze(0).to(device),
                torch.tensor(token_mask).unsqueeze(0).to(device),
                torch.tensor(obs_categories).unsqueeze(0).to(device)
            )
        return obs_embed

    def oracle_obs_embed(pos):
        old_pos = np.copy(world.robot.pos)
        world.robot.pos[:] = pos
        #item = world.items.pop(-1)
        #world.robot.inventory = item
        obs = embed_obs(world.get_obs())

        #world.robot.inventory = None
        world.robot.pos[:] = old_pos
        #world.items.append(item)
        return obs

    target_pos = np.random.random(2)
    #target_pos = world.items[-1].pos
    goal = oracle_obs_embed(target_pos)
    #goal = oracle_obs_embed(None)
    latent_state = model.init_state(embed_obs(last_obs))

    def model_update(latent, obs, action):
        obs_tokens, obs_categories, token_mask = tokenize_obs(obs, pad_to_size=MAX_TOKENS)
        with torch.no_grad():
            obs_emb, latents, obs_reconstruct = model(
                latent.unsqueeze(0).cuda(),
                torch.tensor(obs_tokens).float().unsqueeze(0).cuda(),
                torch.tensor(token_mask).unsqueeze(0).cuda(),
                torch.tensor(obs_categories).unsqueeze(0).cuda(),
                action.unsqueeze(0).cuda()
            )
            return obs_emb[0].cpu(), latents[0].cpu(), obs_reconstruct[0].cpu()

    def clip_actions(actions: torch.tensor):
        #print(actions.shape)
        displacements = actions[..., :2]
        sizes = torch.norm(displacements, dim=-1, keepdim=True)
        #print(displacements.shape, sizes.shape)
        displacements_normalized = displacements * (max_speed / torch.maximum(sizes, torch.tensor(max_speed, device=device)))
        # slice to preserve dimensions
        discrete_action_normalized = torch.round(torch.clamp(actions[..., 2:], min=-1.0, max=1.0))
        return torch.cat((displacements_normalized, discrete_action_normalized), dim=-1)

    planner = CEMPlanner(3, plan_horizon=1, num_candidates=1000, num_elites=100,
                            num_iterations=50, clip_actions=clip_actions, device=device)

    def reward(states, _actions):
        predicted_obs = model.reconstruction(states)
        error = predicted_obs - goal
        return -torch.norm(error, dim=-1)

    def simplify_obs(obs):
        res = "r" + 'c'*len(obs['containers']) + 'i'*len(obs['items'])
        if obs['pickup'] is not None:
            res += 'p'
        if obs['drop'] is not None:
            res += 'd'
        return res

    def render(update=True):
        global latent_state
        if update:
            with torch.no_grad():
                action, actions = planner.plan(latent_state[0], model.openloop_dynamics, reward)
            #print(actions)
            #input()
            obs = world.update(action.cpu().numpy())
            obs_simplify = simplify_obs(obs)

            # Closed loop latent
            latent_state = model.predict_latent(latent_state, embed_obs(obs), clip_actions(action).unsqueeze(0))[:, 1, :]
            # CHEAT: memoryless for now. I think the world model will implode if asked to use even CL states
            #latent_state = model.init_state(embed_obs(obs))
            a = action.cpu().numpy().tolist()
            pos = world.robot.pos
            title = f"Plan world (obs: {obs_simplify}, action: [{a[0]:.3f}, {a[1]:.3f}, {a[2]:.3f}] pos: [{pos[0]:.3f}, {pos[1]:.3f}]"
            title += f" target: [{target_pos[0]:.3f}, {target_pos[1]:.3f}]"
            plotter.set_title(title)

        display = world.render()
        display = 255 - np.mean(display, axis=-1)
        plotter.plot_image_section(display, start_row=0)
        plotter.draw()

    render(update=False)

    import sys, select, termios, time, tty
    def getKey():
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''

        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, _settings)
        return key
    _settings = termios.tcgetattr(sys.stdin)
    render()
    try:
        mode = 0
        while True:
            key = getKey()
            if key == ' ':
                mode = 1 - mode
            if mode == 1:
                render()
            time.sleep(0.05)

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, _settings)
