# Feat gemini
import torch

def slerp(v0: torch.Tensor, v1: torch.Tensor, t: float | torch.Tensor, dot_threshold: float = 0.9995) -> torch.Tensor:
    """
    Spherical linear interpolation between two batches of N-dimensional vectors.

    Args:
        v0: Tensor of shape (batch_size, ..., N)
        v1: Tensor of shape (batch_size, ..., N)
        t: float or Tensor of shape (batch_size, ..., 1) - interpolation parameter [0, 1]
        dot_threshold: Threshold for falling back to linear interpolation to avoid NaN

    Returns:
        Interpolated Tensor of the same shape as v0 and v1
    """
    # Normalize vectors to extract directions and angles
    v0_norm = v0 / torch.linalg.norm(v0, dim=-1, keepdim=True).clamp_min(1e-7)
    v1_norm = v1 / torch.linalg.norm(v1, dim=-1, keepdim=True).clamp_min(1e-7)

    # Compute dot product and clamp to [-1, 1] for numerical stability in acos
    dot = torch.sum(v0_norm * v1_norm, dim=-1, keepdim=True)
    dot = dot.clamp(-1.0, 1.0)

    # Compute the angle (theta) between vectors
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)

    # Calculate interpolation weights
    weight_0 = torch.sin((1.0 - t) * theta) / sin_theta.clamp_min(1e-7)
    weight_1 = torch.sin(t * theta) / sin_theta.clamp_min(1e-7)

    # Standard linear interpolation for highly collinear vectors
    is_collinear = dot > dot_threshold

    res_slerp = weight_0 * v0 + weight_1 * v1
    res_lerp = (1.0 - t) * v0 + t * v1

    # Use torch.where to safely route between slerp and lerp
    return torch.where(is_collinear, res_lerp, res_slerp)
