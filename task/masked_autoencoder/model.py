"""Masked tabular autoencoder: a plain-PyTorch re-implementation of pytorch-widedeep 1.7.0's encoder-decoder
self-supervised pretraining, adapted to padded T x 13 windows.

Source (pytorch_widedeep 1.7.0 wheel, read only; paths relative to the package root):

    models/tabular/self_supervised/_random_obfuscator.py  L14-33   RandomObfuscator: mask ~ Bernoulli(p) per entry
                                                                     (1 = hidden, reconstructed), input * (1 - mask)
    models/tabular/self_supervised/encoder_decoder_model.py L60-70  EncoderDecoderModel._forward: in training the mask
                                                                     is drawn, in eval the mask is all ones
    models/tabular/self_supervised/encoder_decoder_model.py L96-130 decoder = the encoder's widths reversed
    losses.py                                              L908-970 EncoderDecoderLoss
    models/tabular/mlp/_layers.py                          L7-22    dense_layer: [Linear -> act -> (BN) -> Dropout]
                                                                     with linear_first = True
    models/tabular/mlp/_layers.py                          L52-89   MLP: one dense_layer per consecutive width pair
    models/tabular/mlp/tab_mlp.py                          L158-163 TabMlp defaults: relu, dropout 0.1, no batchnorm,
                                                                     linear_first True; L196 encoder = MLP
    models/tabular/mlp/tab_mlp.py                          L223-301 TabMlpDecoder = MLP(hidden + [embed_dim])
    models/tabular/resnet/_layers.py                       L8-55    BasicBlock (Linear no bias -> BN -> LeakyReLU ->
                                                                     Dropout -> Linear -> BN, + identity / resize,
                                                                     LeakyReLU)
    models/tabular/resnet/_layers.py                       L58-90   DenseResnet (input Linear + BN when the input
                                                                     width differs from blocks_dims[0])
    models/tabular/resnet/tab_resnet.py                    L189-190, L239  TabResnet: DenseResnet(blocks_dims, 0.1)
    models/tabular/resnet/tab_resnet.py                    L284-427 TabResnetDecoder: DenseResnet(reversed blocks) +
                                                                     Linear(blocks[-1], embed_dim, bias=False)

With continuous columns only and no `cont_norm_layer`, widedeep's "embeddings" are the raw column values
(`_base_tabular_model.py` L132-147, `cont_norm = nn.Identity()` at L305-306), so the reconstruction target is the input.

Adaptation to padded windows (deviations, also listed in README.md):
  * the window (T x 13) is flattened to 13 T continuous columns; the T-entry row mask is appended as T extra input
    columns (like the ConvAE's mask channel). Mask columns are never obfuscated and never reconstructed.
  * obfuscation is restricted to real rows: hidden = Bernoulli(p) AND real row. Padding cells are zero, never hidden,
    never in the loss and never scored.
  * the encoder sees the obfuscated input (`feed_masked_input`). widedeep 1.7.0's TabMlp / TabResnet path computes
    `masked_x` but feeds `self.encoder(X)` (encoder_decoder_model.py L64-65), i.e. the clean input; only the TabNet path
    (L76-79) uses it. The literal behaviour is kept behind `feed_masked_input = False`.
  * the TabMlpDecoder's reconstruction layer is a plain Linear (`linear_mlp_output`); widedeep's MLP applies ReLU +
    dropout after it too (mlp/_layers.py L20-21), which cannot produce the negative values of z-scored features.
  * the loss divides by per-feature variances pooled over the batch's real cells (13 values) instead of one variance per
    flattened column (13 T values): a column at a late time step can have no real cell in a batch.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn

from task.masked_autoencoder.config import MAEConfig


def dense_layer(inp: int, out: int, p: float) -> nn.Sequential:
    """widedeep `dense_layer` with activation relu, bn False, linear_first True: Linear -> ReLU -> Dropout (if p)."""
    layers: list[nn.Module] = [nn.Linear(inp, out), nn.ReLU(inplace=True)]
    if p != 0:
        layers.append(nn.Dropout(p))
    return nn.Sequential(*layers)


class MLP(nn.Module):
    """widedeep `MLP` (no batchnorm): one `dense_layer` per consecutive pair of `widths`."""

    def __init__(self, widths: list[int], dropout: float) -> None:
        super().__init__()
        self.mlp = nn.Sequential()
        for i in range(1, len(widths)):
            self.mlp.add_module(f"dense_layer_{i - 1}", dense_layer(widths[i - 1], widths[i], dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class BasicBlock(nn.Module):
    """widedeep resnet `BasicBlock` (simplify False)."""

    def __init__(self, inp: int, out: int, dropout: float, resize: nn.Module | None) -> None:
        super().__init__()
        self.resize = resize
        self.lin1 = nn.Linear(inp, out, bias=False)
        self.bn1 = nn.BatchNorm1d(out)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.dp = nn.Dropout(dropout) if dropout > 0.0 else None
        self.lin2 = nn.Linear(out, out, bias=False)
        self.bn2 = nn.BatchNorm1d(out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.leaky_relu(self.bn1(self.lin1(x)))
        if self.dp is not None:
            out = self.dp(out)
        out = self.bn2(self.lin2(out))
        if self.resize is not None:
            identity = self.resize(x)
        out = out + identity
        return self.leaky_relu(out)


class DenseResnet(nn.Module):
    """widedeep `DenseResnet`: optional input projection, then residual blocks through `blocks_dims`."""

    def __init__(self, input_dim: int, blocks_dims: list[int], dropout: float) -> None:
        super().__init__()
        if input_dim != blocks_dims[0]:
            self.dense_resnet = nn.Sequential(
                OrderedDict(
                    [
                        ("lin_inp", nn.Linear(input_dim, blocks_dims[0], bias=False)),
                        ("bn_inp", nn.BatchNorm1d(blocks_dims[0])),
                    ]
                )
            )
        else:
            self.dense_resnet = nn.Sequential()
        for i in range(1, len(blocks_dims)):
            resize = None
            if blocks_dims[i - 1] != blocks_dims[i]:
                resize = nn.Sequential(
                    nn.Linear(blocks_dims[i - 1], blocks_dims[i], bias=False), nn.BatchNorm1d(blocks_dims[i])
                )
            self.dense_resnet.add_module(
                f"block_{i - 1}", BasicBlock(blocks_dims[i - 1], blocks_dims[i], dropout, resize)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense_resnet(x)


class RandomObfuscator(nn.Module):
    """widedeep `RandomObfuscator` restricted to real rows: hidden ~ Bernoulli(p) on cells of real rows only.

    The Bernoulli draw is made on the CPU (as widedeep does: `torch.bernoulli(p * torch.ones(x.shape))`) from the global
    torch generator, which the run seed controls, and then moved to the input's device.
    """

    def __init__(self, p: float) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor, row_mask: torch.Tensor) -> torch.Tensor:
        """(B, T, F) bool: True = hidden (and therefore reconstructed and scored). Never True on a padding row."""
        draw = torch.bernoulli(self.p * torch.ones(x.shape)).to(x.device).bool()
        return draw & row_mask.unsqueeze(-1)


class MaskedTabularAutoencoder(nn.Module):
    """Encoder-decoder over flattened windows.

    `forward(x (B, T, F), row_mask (B, T), hidden (B, T, F) bool or None)` returns (reconstruction (B, T, F), z (B, d)).
    Input columns = the 13 T feature values (hidden and padding cells set to 0) followed by the T row-mask values.
    """

    def __init__(self, settings: MAEConfig) -> None:
        super().__init__()
        self.seq_len, self.n_features = settings.seq_len, settings.n_features
        self.feed_masked_input = settings.feed_masked_input
        self.n_out = self.seq_len * self.n_features
        n_in = self.n_out + self.seq_len
        if settings.encoder == "mlp":
            dims = list(settings.mlp_hidden_dims)
            self.encoder = MLP([n_in] + dims, settings.mlp_dropout)
            rev = dims[::-1]
            if settings.linear_mlp_output:
                self.decoder = nn.Sequential(MLP(rev, settings.mlp_dropout), nn.Linear(rev[-1], self.n_out))
            else:
                self.decoder = MLP(rev + [self.n_out], settings.mlp_dropout)
            self.latent_dim = dims[-1]
        else:
            dims = list(settings.blocks_dims)
            self.encoder = DenseResnet(n_in, dims, settings.blocks_dropout)
            rev = dims[::-1]
            self.decoder = nn.Sequential(
                DenseResnet(rev[0], rev, settings.blocks_dropout), nn.Linear(rev[-1], self.n_out, bias=False)
            )
            self.latent_dim = dims[-1]
        self.masker = RandomObfuscator(settings.masked_prob)

    def inputs(self, x: torch.Tensor, row_mask: torch.Tensor, hidden: torch.Tensor | None) -> torch.Tensor:
        """(B, 14 T) encoder input: features with padding (and hidden cells, if any) zeroed, then the row mask."""
        m = row_mask.to(x.dtype)
        keep = m.unsqueeze(-1)
        if hidden is not None and self.feed_masked_input:
            keep = keep * (~hidden).to(x.dtype)
        return torch.cat([(x * keep).flatten(1), m], dim=1)

    def encode(self, x: torch.Tensor, row_mask: torch.Tensor, hidden: torch.Tensor | None = None) -> torch.Tensor:
        return self.encoder(self.inputs(x, row_mask, hidden))

    def forward(
        self, x: torch.Tensor, row_mask: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x, row_mask, hidden)
        recon = self.decoder(z).view(-1, self.seq_len, self.n_features)
        return recon, z


class EncoderDecoderLoss(nn.Module):
    """widedeep `EncoderDecoderLoss` (losses.py L954-970) over hidden real cells.

    loss = mean over windows of  sum_{hidden cells} (pred - true)^2 / var_f  /  (n hidden cells of the window + eps),
    where var_f is the batch variance (unbiased, as `torch.std(...) ** 2`) of feature f over the batch's real cells;
    a zero variance falls back to |mean_f| and a zero mean to 1, as in widedeep. A window with no hidden cell
    contributes 0 and still counts in the mean (widedeep's eps behaviour).
    """

    def __init__(self, eps: float = 1e-9) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self, x_true: torch.Tensor, x_pred: torch.Tensor, hidden: torch.Tensor, row_mask: torch.Tensor
    ) -> torch.Tensor:
        real = row_mask.unsqueeze(-1).to(x_true.dtype)  # (B, T, 1)
        n_real = real.sum().clamp_min(2.0)
        means = (x_true * real).sum(dim=(0, 1)) / n_real
        var = (((x_true - means) * real) ** 2).sum(dim=(0, 1)) / (n_real - 1.0)
        means = torch.where(means == 0, torch.ones_like(means), means)
        var = torch.where(var == 0, means.abs(), var)
        h = hidden.to(x_true.dtype)
        errors = ((x_pred - x_true) * h) ** 2 / var
        per_window = errors.sum(dim=(1, 2)) / (h.sum(dim=(1, 2)) + self.eps)
        return per_window.mean()


def masked_cell_mse(x_true: torch.Tensor, x_pred: torch.Tensor, cells: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """(sum of squared errors, number of cells) per window over the boolean `cells` (B, T, F)."""
    c = cells.to(x_true.dtype)
    return (((x_pred - x_true) * c) ** 2).sum(dim=(1, 2)), c.sum(dim=(1, 2))


def count_params(module: nn.Module) -> int:
    """Trainable parameters."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
