import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, einsum

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

    def forward(self, x, attn_mask=None, causal=True):
        """
        x : (B, T, D)
        """
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # q, k, v: (B, heads, T, dim_head)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        attn_mask = rearrange(attn_mask, "b x y -> b 1 x y")    # Account for attention heads
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, attn_mask=attn_mask, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


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
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), attn_mask=mask, causal=False)
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
        x = x + self.attn(self.norm1(x), attn_mask=mask, causal=False)
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

    def forward(self, x, mask, c=None):

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
        dim_head=64,
        dropout=0.0,
        #emb_dropout=0.0,
    ):
        super().__init__()
        self.cat_embedding = nn.Parameter(torch.randn(hidden_dim, categories))
        #self.dropout = nn.Dropout(emb_dropout)
        #self.empty_state = nn.Parameter(torch.randn(input_dim))

        self.token_proj = nn.Linear(input_dim, hidden_dim)

        self.transformer = Transformer(
            action_dim,
            hidden_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
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
        B, ntok, _ = x.shape
        T = x.size(1)
        #x = self.dropout(x)
        x = self.token_proj(x)  # B, ntok, d_hidden
        x = x + einsum(self.cat_embedding, categories_onehot.float(), "x c, b n c -> b n x")
        prior_latents = rearrange(prior_latents, "b d -> b 1 d")
        x = torch.cat((prior_latents, x), 1)

        # NOTE: padding acts on the LAST dimension. (left, right)
        mask_pad1 = F.pad(token_mask.int(), (1, 0), "constant", 0)
        attention_mask = einsum(
            torch.ones(ntok+1, device=x.device),
            mask_pad1,
            "d, b n -> b d n"
        )
        # Required since we are doing single-step single-step prediction... no action or state history.
        c = rearrange(c, "b a -> b 1 a")

        # lower right should be 1s whenever there is a nonzero mask token.
        # First column is all 1s, so that everyone can attend to the prior latent
        attention_mask[:, :, 0] = 1
        x = self.transformer(x, attention_mask, c)
        return x[:, -1, :]  # Return last token embedding

