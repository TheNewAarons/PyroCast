"""Focal Loss (Lin et al. 2017, "Focal Loss for Dense Object
Detection") para máscaras binarias desbalanceadas -- la elección sobre
BCE ponderada (`pos_weight` de BCEWithLogitsLoss):

BCE ponderada corrige el desbalance de FRECUENCIA de clase (un peso
escalar fijo para la clase positiva), pero sigue penalizando por igual
cada negativo, sin importar qué tan fácil sea. En una máscara de fuego
real, la gran mayoría de los píxeles son negativos TRIVIALES (lejos de
cualquier fuego, el modelo aprende rápido a predecirlos con alta
confianza) -- BCE ponderada sigue acumulando pérdida sobre ellos,
diluyendo el gradiente útil de los píxeles difíciles (el borde del
frente de fuego, los pocos positivos reales).

Focal loss agrega un término de "focusing", `(1-p_t)^gamma`, que baja
la pérdida de cualquier predicción ya confiada y correcta (sea positiva
o negativa) -- no solo corrige la frecuencia de clase con `alpha`, sino
que concentra el gradiente en los píxeles donde el modelo todavía se
equivoca o duda. Ver `test_focal_loss_down_weights_easy_negatives_more_than_plain_bce`
para la propiedad verificada, y docs/model-card.md para más contexto.

FL(p_t) = -alpha_t * (1-p_t)^gamma * log(p_t)
    p_t = p si target=1, (1-p) si target=0    (p = sigmoid(logit))
    alpha_t = alpha si target=1, (1-alpha) si target=0

Defaults: alpha=0.8 (la clase "fuego" es la minoría, se le da más
peso), gamma=2.0 (el valor por defecto del paper original)."""
import torch
from torch import nn


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.8, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        focal_term = (1 - p_t) ** self.gamma
        loss = alpha_t * focal_term * bce
        result: torch.Tensor = loss.mean()
        return result
