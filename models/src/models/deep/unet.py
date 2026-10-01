"""U-Net 2D pequeña (pocos bloques) que recibe el tensor de UN solo día
(canales estáticos + dinámicos del día actual + fire_mask del día
actual, mismo orden que features.dataset.assemble.CHANNEL_ORDER) y
predice, por celda, el LOGIT de probabilidad de fuego del día
SIGUIENTE -- no una probabilidad ya aplicada con sigmoid (ver
docs/decisions.md: BCEWithLogitsLoss/FocalLoss esperan logits, más
estable numéricamente que sigmoid+log por separado).

GroupNorm, no BatchNorm: este proyecto entrena con batch_size=1 por
defecto (ver docs/model-card.md -- sin GPU, eventos reales de tamaño
variable), donde las estadísticas de BatchNorm no tienen sentido.
GroupNorm normaliza dentro de una sola muestra, funciona igual a
cualquier batch_size. Requiere que cada nivel de canales sea divisible
por num_groups (8, fijo) -- por eso `base_channels` debe ser múltiplo
de 8.

Tamaño arbitrario de entrada: los eventos reales de PyroCast tienen el
tamaño de su propio bbox (features/grid/grid.py::WorkGrid), nunca
garantizado divisible por 2**depth. El camino de subida rellena
(pad) cada mapa de activación subido al tamaño EXACTO de su conexión
de salto antes de concatenar -- la técnica estándar (usada por la
mayoría de las implementaciones modernas de U-Net) para tolerar
cualquier tamaño de entrada sin recortar información."""
import torch
import torch.nn.functional as F
from torch import nn

from models.deep.normalization import normalize_inputs

_NUM_GROUPS = 8


class _DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_NUM_GROUPS, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_NUM_GROUPS, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = self.block(x)
        return result


class _Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool_conv = nn.Sequential(nn.MaxPool2d(2), _DoubleConv(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = self.pool_conv(x)
        return result


class _Up(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = _DoubleConv(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        # pad x al tamaño EXACTO de skip -- ver docstring del módulo.
        diff_h = skip.shape[2] - x.shape[2]
        diff_w = skip.shape[3] - x.shape[3]
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])
        merged = torch.cat([skip, x], dim=1)
        result: torch.Tensor = self.conv(merged)
        return result


class SmallUNet(nn.Module):
    def __init__(
        self, in_channels: int = 11, base_channels: int = 16, depth: int = 3,
        input_norm: str = "none",
    ) -> None:
        super().__init__()
        normalize_inputs(torch.zeros(1, in_channels, 1, 1), input_norm)  # valida el modo
        self.input_norm = input_norm
        if base_channels % _NUM_GROUPS != 0:
            raise ValueError(
                f"base_channels debe ser múltiplo de {_NUM_GROUPS} (GroupNorm) -- "
                f"recibido {base_channels}."
            )
        self.depth = depth
        self.in_conv = _DoubleConv(in_channels, base_channels)

        channels = [base_channels * (2**i) for i in range(depth + 1)]
        self.downs = nn.ModuleList(
            [_Down(channels[i], channels[i + 1]) for i in range(depth)]
        )
        self.ups = nn.ModuleList(
            [_Up(channels[i + 1], channels[i], channels[i]) for i in reversed(range(depth))]
        )
        self.out_conv = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # normalización fija de entradas (models/deep/normalization.py); sin
        # parámetros ni buffers: el state_dict no cambia.
        x = normalize_inputs(x, self.input_norm)
        skips = [self.in_conv(x)]
        for down in self.downs:
            skips.append(down(skips[-1]))

        y = skips[-1]
        for i, up in enumerate(self.ups):
            skip = skips[-2 - i]
            y = up(y, skip)

        result: torch.Tensor = self.out_conv(y)
        return result
