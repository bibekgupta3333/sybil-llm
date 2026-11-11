"""The model: the T = 48 ConvAutoencoder of `task/autoencoder` (imported, not copied).

14 x 48 (13 features + mask) -> z (16) -> 13 x 48 reconstruction, 71,741 trainable parameters; masked MSE over real
rows. `ConvAutoencoder(settings)` reads `n_features`, `seq_len` and `latent_dim` of this package's config.
"""

from __future__ import annotations

from task.autoencoder.model import ConvAutoencoder, count_params, masked_mse, window_errors

__all__ = ["ConvAutoencoder", "count_params", "masked_mse", "window_errors"]
