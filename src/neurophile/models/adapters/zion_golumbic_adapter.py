"""
neurophile.models.adapters.zion_golumbic_adapter
================================================

Adapter wrapping the Zion-Golumbic cross-attention model for AAD.
"""

import logging

from neurophile.models.core.base_aad_model import BaseAADModel, _require_torch

logger = logging.getLogger(__name__)

# ACL shim — guard the external import
_ZG_EXTERNAL_AVAILABLE = False
try:
    from external_libs.zion_golumbic.model import CrossAttentionAAD
    _ZG_EXTERNAL_AVAILABLE = True
    logger.info("ZionGolumbicAdapter: using real CrossAttentionAAD from external_libs.")
except ImportError:
    pass

def _build_fallback_cross_attention(num_eeg_channels: int):
    """Real implementation of Cross-Attention with temporal convolutions and positional encodings."""
    _require_torch()
    import torch
    import torch.nn as nn
    
    class FallbackCrossAttention(nn.Module):
        def __init__(self, num_eeg_channels):
            super().__init__()
            # Use Conv1d to extract temporal features (kernel_size=15 covers ~250ms at 64Hz)
            self.eeg_conv = nn.Conv1d(num_eeg_channels, 32, kernel_size=15, padding=7)
            self.env_conv = nn.Conv1d(1, 32, kernel_size=15, padding=7)
            
            # Learnable positional encoding so the model knows which time step it is looking at!
            self.pos_emb = nn.Parameter(torch.randn(1, 1024, 32) * 0.02)
            
            self.norm1 = nn.LayerNorm(32)
            self.norm2 = nn.LayerNorm(32)
            
            self.attn = nn.MultiheadAttention(embed_dim=32, num_heads=4, batch_first=True)
            self.dropout = nn.Dropout(0.2)
            
            self.fc = nn.Sequential(
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(16, 1)
            )
            
        def forward(self, eeg, env):
            # Conv1d expects (Batch, Channels, Time)
            eeg_c = self.eeg_conv(eeg.transpose(1, 2)).transpose(1, 2)
            env_c = self.env_conv(env.transpose(1, 2)).transpose(1, 2)
            
            # Add positional encodings
            T = eeg_c.size(1)
            eeg_c = eeg_c + self.pos_emb[:, :T, :]
            env_c = env_c + self.pos_emb[:, :T, :]
            
            # Apply LayerNorm BEFORE attention (Pre-LN Transformer standard)
            eeg_norm = self.norm1(eeg_c)
            env_norm = self.norm2(env_c)
            
            # Query: env, Key/Value: eeg
            attn_out, _ = self.attn(env_norm, eeg_norm, eeg_norm)
            attn_out = self.dropout(attn_out)
            
            # Crucial Residual Connection!
            out_seq = env_c + attn_out
            out_seq = torch.relu(out_seq)
            
            # Pool over time and project
            pooled = out_seq.mean(dim=1)
            return self.fc(pooled)
            
    return FallbackCrossAttention(num_eeg_channels)

def _make_zg_adapter_class():
    _require_torch()
    import torch.nn as nn

    class ZionGolumbicAdapter(nn.Module, BaseAADModel):
        """Neurophile adapter for the Zion-Golumbic Cross-Attention Model."""
        name = "zion_golumbic_cross_attention"

        def __init__(self, num_eeg_channels=64, audio_sampling_rate=64, use_external=False):
            super().__init__()
            self.num_eeg_channels = num_eeg_channels
            self.audio_sampling_rate = audio_sampling_rate
            self._using_external = use_external
            
            if use_external and not _ZG_EXTERNAL_AVAILABLE:
                raise ImportError(
                    "ZionGolumbicAdapter: external_libs.zion_golumbic not found — "
                    "did you run clone_author_repos.sh?"
                )
                
            if use_external:
                # TODO (implementer): Adjust constructor arguments
                self.backend_model = CrossAttentionAAD(n_channels=num_eeg_channels)
            else:
                self.backend_model = _build_fallback_cross_attention(num_eeg_channels)

        def forward(self, eeg_tensor, audio_envelope_tensor):
            if self._using_external:
                # TODO (implementer): Translate Neurophile tensors to ZG's expected format
                pass
            return self.backend_model(eeg_tensor, audio_envelope_tensor)

    return ZionGolumbicAdapter

ZionGolumbicAdapter = _make_zg_adapter_class()
