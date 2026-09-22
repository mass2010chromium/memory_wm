"""
Observation reconstruction benchmark feat. gemini
"""

import argparse
import json
import os
import numpy as np
import torch
import tqdm

from memory_wm.module import Predictor
from env_2d import tokenize_obs, World2d, MAX_TOKENS
from dump_embeddings import dump_embeddings

def load_model(model_config, checkpoint_path):
    data = torch.load(checkpoint_path, weights_only=True)
    model = Predictor(**model_config).cuda()
    model.load_state_dict(data['model_state'])
    model.eval()
    return model

def model_update(model, latent, obs, action):
    obs_tokens, obs_categories, token_mask = tokenize_obs(obs, pad_to_size=MAX_TOKENS)
    with torch.no_grad():
        obs_emb, latents, obs_reconstruct = model(
            latent.unsqueeze(0).cuda(),
            torch.tensor(obs_tokens).float().unsqueeze(0).cuda(),
            torch.tensor(token_mask).unsqueeze(0).cuda(),
            torch.tensor(obs_categories).unsqueeze(0).cuda(),
            torch.tensor(action).float().unsqueeze(0).cuda()
        )
        return obs_emb[0].cpu(), latents[0].cpu(), obs_reconstruct[0].cpu()

def main():
    parser = argparse.ArgumentParser(description="Observation prediction benchmarking tool")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint folder")
    parser.add_argument("--embeddings-folder", type=str, required=True, help="Path to precomputed reference embeddings folder (embeddings/42)")
    parser.add_argument("--world-data", type=str, default="world.json", help="Path to world data JSON")
    parser.add_argument("--model-config", type=str, default="config/model_config.json", help="Path to model config JSON")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Path to save output JSON")
    parser.add_argument("--seeds", type=int, default=100, help="Number of random seeds/worlds to run")
    parser.add_argument("--resolution", type=int, default=100, help="Resolution of grids to dump")
    parser.add_argument("--mode", type=str, choices=["closed_loop", "open_loop", "1_step"], default="closed_loop", 
                        help="Dynamics model path to use (closed_loop=latents[-1], open_loop=latents[-2], 1_step=init_state)")
    
    args = parser.parse_args()

    # Load configs and model
    with open(args.world_data, "r") as jf:
        world_data = json.load(jf)
        
    with open(args.model_config, "r") as jf:
        config = json.load(jf)
        
    embeddings_folder = args.embeddings_folder

    if args.checkpoint is not None:
        checkpoint = os.path.abspath(args.checkpoint)
        model = load_model(config, checkpoint)
    else:
        checkpoint = None
        model = None

    def ensure_exists(world, seed, model, checkpoint):
        target_path = os.path.join(embeddings_folder, str(seed), "checkpoint.txt")
        if not os.path.exists(target_path):
            if model is None:
                print("Could not initialize embeddings without specified checkpoint. Specify checkpoint manually with --checkpoint")
                exit(1)
            embeddings = dump_embeddings(world, seed, config, embeddings_folder, model, checkpoint, steps=args.resolution)
            return model, checkpoint, embeddings
        with open(target_path, "r") as infile:
            target_checkpoint = infile.read().strip()
            if model is None:
                print("Initializing model from", target_path)
                model = load_model(config, target_checkpoint)
                checkpoint = target_checkpoint

                embeddings = np.load(os.path.join(embeddings_folder, str(seed), "embeddings.npy"))
                if embeddings.shape[0] != args.resolution:
                    print(f"WARN: Reloading seed {seed}, resolution does not match. Overwriting")
                    embeddings = dump_embeddings(world, seed, config, embeddings_folder, model, checkpoint, steps=args.resolution)
            elif checkpoint != target_checkpoint:
                print(f"WARN: Reloading seed {seed}, checkpoint does not match. Overwriting")
                embeddings = dump_embeddings(world, seed, config, embeddings_folder, model, checkpoint, steps=args.resolution)
            else:
                embeddings = np.load(os.path.join(embeddings_folder, str(seed), "embeddings.npy"))
        return model, checkpoint, embeddings
    
    results = []
    step_limit = 150 # Failsafe limit to prevent infinite loops

    for seed in tqdm.trange(args.seeds):
        # Initialize world and starting point
        np.random.seed(seed)
        world = World2d(world_data)
        world.reset()
        init_obs = world.update([0.0, 0.0, 0.0])

        model, checkpoint, precomputed_embeddings = ensure_exists(world, seed, model, checkpoint)
        
        # Pick a target point uniformly in [0, 1] x [0, 1]
        target = np.random.uniform(0, 1, 2)

        max_movement = 0.25
        delta = target - world.robot.pos
        dist = np.linalg.norm(delta)
        if dist > max_movement:
            delta = delta * (max_movement / dist)
        target = world.robot.pos + delta

        obs_tokens, obs_categories, token_mask = tokenize_obs(init_obs, pad_to_size=MAX_TOKENS)
        
        with torch.no_grad():
            init_obs_embed = model.embed_obs(
                torch.tensor(obs_tokens).float().unsqueeze(0).cuda(),
                torch.tensor(token_mask).unsqueeze(0).cuda(),
                torch.tensor(obs_categories).unsqueeze(0).cuda()
            )
            prior_latent = model.init_state(init_obs_embed[0]).cpu()
            
        max_step_size = world.robot.max_speed
        step_count = 0
        
        obs_err = 0.0
        pos_err = 0.0

        trajectory = []
        pred_trajectory = []
        # Move robot towards target
        while step_count < step_limit:
            delta = target - world.robot.pos
            dist = np.linalg.norm(delta)

            # Reached target
            if dist < 1e-3:
                break
                
            # Normalize vector to max_step_size
            if dist > max_step_size:
                delta = delta * (max_step_size / dist)
                
            action = [*delta, 0.0]  # Ignoring third item as requested
            
            obs_new = world.update(action)
            obs_emb, latents, obs_reconstruct = model_update(model, prior_latent, obs_new, action)
            
            # Advance the latent state based on specified mode
            if args.mode == "closed_loop":
                prior_latent = latents[-1]
            elif args.mode == "open_loop":
                prior_latent = latents[-2]
            elif args.mode == "1_step":
                with torch.no_grad():
                    prior_latent = model.init_state(obs_emb.cuda()).cpu()

            # Coordinate map divided by 100 to map [0-100] grid to [0-1] coordinates
            distances = np.linalg.norm(obs_reconstruct.numpy() - precomputed_embeddings, axis=-1)
            max_position = np.array(np.unravel_index(np.argmin(distances), distances.shape)) / args.resolution
            step_count += 1
            trajectory.append(world.robot.pos.tolist())
            pred_trajectory.append(max_position.tolist())
            
        # At the end of the movement sequence, calculate final metrics
        obs_err = (obs_emb - obs_reconstruct).pow(2).mean().item()

        pos_err = float(np.linalg.norm(world.robot.pos - max_position))

        results.append({
            "seed": seed,
            "target": target.tolist(),
            "final_pos": world.robot.pos.tolist(),
            "steps": step_count,
            "obs_error": obs_err,
            "trajectory": trajectory,
            "pred_trajectory": pred_trajectory,
            "pos_error": pos_err
        })
        
    with open(args.output, "w") as f:
        json.dump({
            "checkpoint": checkpoint,
            "mode": args.mode,
            "runs": results
        }, f, indent=4)
        
    print(f"Benchmarked {args.seeds} seeds in '{args.mode}' mode.")
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
