"""Identity encoder (Section IV-B): five strided 1-D convolutions, a temporal mean, and a linear embedding.

The transient view enters as two I/Q channels tiled to 16384 samples. conv7 and four conv5 layers (stride 2, group
normalization with 8 groups, SiLU) widen it to 32, 64, 128, 128, and 128 channels and halve its length down to 512
samples; a temporal mean (avg_t) and a linear layer (FC) give a 64-dimensional burst embedding on the unit sphere.
A training-only projector 64-256-256 feeds the VICReg term. The two encoders f_T and f_A share this architecture
and have separate weights; the module names match the paper's checkpoints, so their state dicts load directly.
"""
import torch
from torch import nn

EPS = 1e-12
WINDOW = 16384


class IdentityEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        blocks = []
        for c_in, c_out, kernel, stride, pad in ((2, 32, 7, 2, 3), (32, 64, 5, 2, 2), (64, 128, 5, 2, 2),
                                                 (128, 128, 5, 2, 2), (128, 128, 5, 2, 2)):
            blocks += [nn.Conv1d(c_in, c_out, kernel, stride=stride, padding=pad, bias=True),
                       nn.GroupNorm(8, c_out, eps=1e-5, affine=True), nn.SiLU(inplace=False)]
        self.encoder = nn.Sequential(*blocks)
        self.embedding = nn.Linear(128, 64, bias=True)
        self.projector = nn.Sequential(nn.Linear(64, 256), nn.LayerNorm(256, eps=1e-5), nn.SiLU(inplace=False),
                                       nn.Linear(256, 256))

    def forward(self, iq):
        """iq: float32 [B, 2, 16384] (or [B, 2, 256] WiFi preambles). Returns z, the unit-norm e, norm, valid."""
        z = self.embedding(self.encoder(iq).mean(dim=-1))
        norm = torch.linalg.vector_norm(z, dim=1)
        valid = torch.isfinite(z).all(dim=1) & torch.isfinite(norm) & (norm > EPS)
        return {'z': z, 'e': z / norm.clamp_min(EPS)[:, None], 'norm': norm, 'valid': valid}


class CosineHead(nn.Module):
    """Training-only cosine classifier: cosine of the unit embedding to one normalized proxy per identity."""

    def __init__(self, n_classes):
        super().__init__()
        self.weight = nn.Parameter(torch.randn((n_classes, 64), dtype=torch.float32) * .02)

    def forward(self, e):
        return e @ (self.weight / torch.linalg.vector_norm(self.weight, dim=1, keepdim=True)).T


def load_checkpoint(path, map_location='cpu'):
    """Load a paper checkpoint (a dict whose 'model' entry is the encoder state dict)."""
    model = IdentityEncoder()
    model.load_state_dict(torch.load(path, map_location=map_location, weights_only=True)['model'], strict=True)
    return model.eval()


@torch.inference_mode()
def unit_embedding(model, view, windows, device='cpu'):
    """Unit embedding z_m(u) (eq. (5)): mean of the burst embeddings of one recording, normalized.

    model: an IdentityEncoder; view: signal.transient_view (f_T) or signal.channel_view (f_A); windows: array
    [bursts, 2, samples]. Returns (embedding [64] or NaNs, valid flag).
    """
    out = model(view(torch.as_tensor(windows, dtype=torch.float32, device=device)))
    if not bool(out['valid'].all()):
        return torch.full((64,), float('nan')).numpy(), False
    m = out['e'].double().mean(0)
    return (m / m.norm()).cpu().numpy(), True
