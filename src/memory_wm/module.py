import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, einsum

# Copy from: https://colab.research.google.com/drive/11SKfzvMotuvvXNqY9qBpsD2RQX1PK7rP
# With minor edits for my readability.
# NOTE that unlike in the RoPE paper, the pairs here are stored "across the midpoint":
#      (x_0, x_d/2), (x_1, x_d/2+1)...
class RotaryPositionalEmbeddings(nn.Module):

  def __init__(self, d: int, base: int = 100):
    super().__init__()
    self.base = base
    self.d = d
    self.cos_cached = None
    self.sin_cached = None

  def _build_cache(self, x: torch.Tensor):

    if self.cos_cached is not None and x.shape[0] <= self.cos_cached.shape[0]:
      return

    seq_len = x.shape[0]

    # THETA = 10,000^(-2*i/d) or 1/10,000^(2i/d)
    theta = 1. / (self.base ** (torch.arange(0, self.d, 2).float() / self.d)).to(x.device)

    #Position Index -> [0,1,2...seq-1]
    seq_idx = torch.arange(seq_len, device=x.device).float().to(x.device)

    #Calculates m*(THETA) = [ [0, 0...], [THETA_1, THETA_2...THETA_d/2], ... [seq-1*(THETA_1), seq-1*(THETA_2)...] ]
    idx_theta = torch.einsum('n,d->nd', seq_idx, theta)

    # [THETA_1, THETA_2...THETA_d/2] -> [THETA_1, THETA_2...THETA_d]
    idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)


    #Cache [cosTHETA_1, cosTHETA_2...cosTHETA_d], [sinTHETA_1, sinTHETA_2...sinTHETA_d]
    self.cos_cached = idx_theta2.cos()[:, None, None, :]
    self.sin_cached = idx_theta2.sin()[:, None, None, :]

  def _neg_half(self, x: torch.Tensor):

    d_2 = self.d // 2 #

    # [x_1, x_2,...x_d] -> [-x_d/2, ... -x_d, x_1, ... x_d/2]
    return torch.cat([-x[:, :, :, d_2:], x[:, :, :, :d_2]], dim=-1)


  def forward(self, x: torch.Tensor):

    self._build_cache(x)

    neg_half_x = self._neg_half(x)

    # [x_1*cosTHETA_1 - x_d/2*sinTHETA_d/2, ....]
    return (x * self.cos_cached[:x.shape[0]]) + (neg_half_x * self.sin_cached[:x.shape[0]])

# Copying from LeWorldModel codebase
class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!)"""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (B, D)
        """
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean() # average over projections and time

class FeedForward(nn.Module):
    """FeedForward network used in Transformers"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with causal masking"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )
        self.rope = RotaryPositionalEmbeddings(d=dim_head)

    def forward(self, x, attn_mask=None, causal=True):
        """
        x : (B, T, D)
        """
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # q, k, v: (B, heads, T, dim_head)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        q = self.rope(q)
        k = self.rope(q)
        if attn_mask is not None:
            attn_mask = rearrange(attn_mask, "b x y -> b 1 x y")    # Account for attention heads
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, attn_mask=attn_mask, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return x + self.to_out(out)


def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift

class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c, mask):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        if mask is not None:
            x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), attn_mask=mask, causal=False)
        else:
            x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), causal=True)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    """Standard Transformer block"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, mask):
        if mask is not None:
            x = x + self.attn(self.norm1(x), attn_mask=mask, causal=False)
        else:
            x = x + self.attn(self.norm1(x), causal=True)
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Standard Transformer with support for AdaLN-zero blocks"""

    def __init__(
        self,
        cond_dim,
        hidden_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.cond_proj = (
            nn.Linear(cond_dim, hidden_dim)
            if cond_dim != hidden_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x, mask=None, c=None):

        if c is not None and hasattr(self, "cond_proj"):
            c = self.cond_proj(c)

        for block in self.layers:
            x = block(x, mask) if isinstance(block, Block) else block(x, c, mask)

        x = self.norm(x)
        return x


class MLP(nn.Module):
    """Simple MLP with optional normalization and activation"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """
        x: (B*T, D)
        """
        return self.net(x)


class Predictor(nn.Module):
    """Predictor for next-step embedding prediction."""

    def __init__(
        self,
        *,
        categories,
        depth,
        heads,
        mlp_dim,
        action_dim,
        input_dim,
        hidden_dim,
        obs_dim,
        dim_head=64,
        dropout=0.0,
        #emb_dropout=0.0,
    ):
        super().__init__()
        self.cat_embedding = nn.Parameter(torch.randn(obs_dim, categories))
        #self.dropout = nn.Dropout(emb_dropout)
        #self.empty_state = nn.Parameter(torch.randn(input_dim))

        self.obs_proj = nn.Linear(input_dim, obs_dim)

        self.obs_embedder = Transformer(
            obs_dim,
            obs_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=Block,
        )
        self.dynamics = Transformer(
            action_dim,
            hidden_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )
        self.reconstruction = MLP(
            hidden_dim,
            hidden_dim * 2,
            obs_dim
        )
        self.init_embedder = MLP(
            obs_dim, 
            hidden_dim * 2,
            hidden_dim
        )

    def forward(self, prior_latents, x, token_mask, categories_onehot, c):
        """
        prior_latents: (B, d_hidden)
        x: (B, ntok, d_obs)
        token_mask: (B, ntok)
        categories_onehot: (B, ntok, ncat)
        c: (B, act_dim)

        Return: (B, d_hidden)
        """
        obs_embedding = self.embed_obs(x, token_mask, categories_onehot)

        latents = self.predict_latent(prior_latents, obs_embedding, c)

        obs_reconstruct = self.reconstruction(latents[:, 0, :])
        return obs_embedding, latents, obs_reconstruct

    def predict_latent(self, prior_latents, obs_embedding, action):
        # Required since we are doing single-step single-step prediction... no action or state history.
        B, D = prior_latents.shape
        c = rearrange(action, "b a -> b 1 a") # For conditionalblock
        prior_latents = rearrange(prior_latents, "b d -> b 1 d")

        obs_token = rearrange(obs_embedding, "b d -> b 1 d")
        full_obs_token = torch.zeros(B, 1, D, dtype=prior_latents.dtype, device=prior_latents.device)
        full_obs_token[:, :, :obs_token.shape[-1]] = obs_token

        history_and_obs = torch.cat((prior_latents, full_obs_token), 1)

        # Token 0 is the open loop latent (evolved with conditioning c)
        # Token 1 is the closed loop latent (evolved with conditioning and obs embedding by causal attention)
        return self.dynamics(history_and_obs, mask=None, c=c)


    def embed_obs(self, x, token_mask, categories_onehot):
        B, ntok, _ = x.shape
        #x = self.dropout(x)
        x = self.obs_proj(x)  # B, ntok, d_hidden
        # index into category embeddings via matrix multiply
        x = x + einsum(self.cat_embedding, categories_onehot.float(), "x c, b n c -> b n x")
        #x = torch.cat((prior_latents, x), 1)   # For conditionalblock
        # NOTE: padding acts on the LAST dimension. (left, right)
        #mask_pad1 = F.pad(token_mask.int(), (1, 0), "constant", 0)

        # Matrices of full attention for all observation tokens.
        attention_mask = einsum(
            #torch.ones(ntok+1, device=x.device),   # For conditionalblock
            torch.ones(ntok, device=x.device),
            token_mask,
            "d, b n -> b d n"
        )
        # Observation tokens are left aligned. Poor design decision, maybe?
        token_count = einsum(token_mask, 'b n -> b')    # Row sum
        x = self.obs_embedder(x, attention_mask)
        return x[torch.arange(x.size(0)), token_count-1]  # Return last token embedding
        

    def init_state(self, obs_embed):
        return self.init_embedder(obs_embed)
