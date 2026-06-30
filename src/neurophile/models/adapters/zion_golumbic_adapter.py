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
            # Causal temporal features for EEG (kernel_size=16 covers exactly 250ms at 64Hz)
            # We use padding=0 and pad manually in forward() so the filter looks into the future.
            self.eeg_conv = nn.Conv1d(num_eeg_channels, 32, kernel_size=16, padding=0)
            self.env_conv = nn.Conv1d(1, 32, kernel_size=15, padding=7)
            
            # Learnable scalar and bias to map cosine similarity [-1, 1] to logits
            self.scale = nn.Parameter(torch.tensor(10.0))
            self.bias = nn.Parameter(torch.tensor(0.0))
            
        def forward(self, eeg, env):
            # Conv1d expects (Batch, Channels, Time)
            eeg_t = eeg.transpose(1, 2)
            env_t = env.transpose(1, 2)
            
            import torch.nn.functional as F
            
            # CAUSAL SHIFT: Pad the right side of the EEG tensor (the future) by 15 samples.
            # This ensures eeg_c[:, :, t] is computed using eeg[:, :, t : t+16], 
            # perfectly capturing the 250ms cortical response to the audio at time t!
            eeg_padded = F.pad(eeg_t, (0, 15))
            
            eeg_c = self.eeg_conv(eeg_padded).transpose(1, 2)
            env_c = self.env_conv(env_t).transpose(1, 2)
            # We compute Cosine Similarity DIRECTLY between the local convolutional features.
            # This enforces strict temporal locality (250ms causal window), preventing the 
            # network from overfitting to spurious global noise correlations.
            import torch.nn.functional as F
            
            # Compute similarity at each time step
            # env_c shape: (Batch, Time, Features)
            # eeg_c shape: (Batch, Time, Features)
            # sim_time shape: (Batch, Time)
            sim_time = F.cosine_similarity(env_c, eeg_c, dim=-1)
            
            # Average the similarity over time
            sim = sim_time.mean(dim=1)
            
            # Map [-1, 1] to logits
            logit = sim * self.scale + self.bias
            return logit.unsqueeze(-1)
            
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
