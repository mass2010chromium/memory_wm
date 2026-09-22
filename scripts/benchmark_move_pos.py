import argparse
import json
import os
import numpy as np
import torch

# Citation: CEMPlanner imported directly from the user's provided planner.py
from planner import CEMPlanner
from memory_wm.module import Predictor
from env_2d import tokenize_obs, World2d, MAX_TOKENS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_model(checkpoint_path, config_path, device):
    """Loads the Predictor model from a checkpoint."""
    with open(config_path, "r") as jf:
        model_config = json.load(jf)
        
    data = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = Predictor(**model_config).to(device)
    model.load_state_dict(data['model_state'])
    model.eval()
    return model

def run_benchmark():
    parser = argparse.ArgumentParser(description="Run CEM Planner Benchmarks")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--max_steps", type=int, default=100, help="Maximum steps per trial")
    parser.add_argument("--error", type=float, default=0.05, help="Acceptable error threshold for success")
    parser.add_argument("--trials", type=int, required=True, help="Number of trials to run")
    parser.add_argument("--out", type=str, default="benchmark_results.json", help="Output JSON file path")
    parser.add_argument("--world_cfg", type=str, default="world.json", help="Path to world configuration")
    parser.add_argument("--model_cfg", type=str, default="config/model_config.json", help="Path to model config")
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model and world config
    model = load_model(args.checkpoint, args.model_cfg, device)
    with open(args.world_cfg, "r") as jf:
        world_data = json.load(jf)

    # Helper functions for the planner
    def embed_obs(obs):
        obs_tokens, obs_categories, token_mask = tokenize_obs(obs, pad_to_size=MAX_TOKENS)
        with torch.no_grad():
            obs_embed = model.embed_obs(
                torch.tensor(obs_tokens).float().unsqueeze(0).to(device),
                torch.tensor(token_mask).unsqueeze(0).to(device),
                torch.tensor(obs_categories).unsqueeze(0).to(device)
            )
        return obs_embed

    def oracle_obs_embed(world_env, pos):
        old_pos = np.copy(world_env.robot.pos)
        world_env.robot.pos[:] = pos
        obs = embed_obs(world_env.get_obs())
        world_env.robot.pos[:] = old_pos
        return obs

    results = []
    successes = 0

    for trial in range(args.trials):
        np.random.seed(trial)
        
        world = World2d(world_data)
        last_obs = world.reset()
        max_speed = world.robot.max_speed
        init_pos = np.copy(world.robot.pos)
        
        # Generate uniform random target in [0, 1] x [0, 1]
        target_pos = np.random.uniform(0.0, 1.0, size=2)
        max_movement = 0.25
        delta = target_pos - world.robot.pos
        dist = np.linalg.norm(delta)
        if dist > max_movement:
            delta = delta * (max_movement / dist)
        target_pos = world.robot.pos + delta

        goal_embed = oracle_obs_embed(world, target_pos)

        def clip_actions(actions: torch.Tensor):
            displacements = actions[..., :2]
            sizes = torch.norm(displacements, dim=-1, keepdim=True)
            max_speed_tensor = torch.tensor(max_speed, device=device)
            displacements_normalized = displacements * (max_speed / torch.maximum(sizes, max_speed_tensor))
            discrete_action_normalized = torch.round(torch.clamp(actions[..., 2:], min=-1.0, max=1.0))
            return torch.cat((displacements_normalized, discrete_action_normalized), dim=-1)

        def reward_fn(states, _actions):
            predicted_obs = model.reconstruction(states)
            error_val = predicted_obs - goal_embed
            return -torch.norm(error_val, dim=-1)

        planner = CEMPlanner(
            action_dim=3, plan_horizon=6, num_candidates=1000, 
            num_elites=100, num_iterations=5, clip_actions=clip_actions, device=device
        )

        trial_success = False
        steps_taken = 0

            
        # Initial obs
        latent_state = model.init_state(embed_obs(last_obs))
        for step in range(args.max_steps):
            steps_taken += 1
            
            with torch.no_grad():
                action = planner.plan(latent_state[0], model.openloop_dynamics, reward_fn)
            
            last_obs = world.update(action.cpu().numpy())
            
            # Re-initialize latent state at each step (Memoryless approach based on planner.py)
            #latent_state = model.init_state(embed_obs(last_obs))
            # Get proper CL latent
            latent_state = model.predict_latent(latent_state, embed_obs(obs), clip_actions(action).unsqueeze(0))[:, 1, :]
            
            # Check actual physical distance to target
            current_dist = np.linalg.norm(world.robot.pos - target_pos)
            if current_dist <= args.error:
                trial_success = True
                break
                
        if trial_success:
            successes += 1
            
        results.append({
            "success": trial_success,
            "world_seed": trial,
            "init_pos": init_pos.tolist(),
            "target": target_pos.tolist(),
            "steps_taken": steps_taken
        })
        
        print(f"Trial {trial + 1}/{args.trials} - Success: {trial_success}, Steps: {steps_taken}")

    # Compile and save JSON
    output_data = {
        "global_success_rate": successes / args.trials,
        "checkpoint": args.checkpoint,
        "tolerance": args.error,
        "max_steps": args.max_steps,
        "runs": results
    }

    with open(args.out, "w") as f:
        json.dump(output_data, f, indent=4)
        
    print(f"Benchmark complete. Success rate: {output_data['global_success_rate'] * 100:.2f}%. Results saved to {args.out}")

if __name__ == "__main__":
    run_benchmark()
