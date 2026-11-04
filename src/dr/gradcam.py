"""Grad-CAM implementado do zero (sem dependências extras)."""
import cv2
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    """`target_layer` não pode ter a saída alterada in-place pela camada seguinte, pois o
    hook de backward não aceita isso. `model.features[-1]` da EfficientNet é seguro: depois
    dele vem o avgpool. Uma instância guarda estado e não deve ser usada em paralelo."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model, self.acts, self.grads = model.eval(), None, None
        self._hooks = [
            target_layer.register_forward_hook(lambda m, i, o: setattr(self, "acts", o.detach())),
            target_layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "grads", go[0].detach())),
        ]

    def remove(self) -> None:
        """Remove os hooks; sem isso, cada nova instância acumula hooks no modelo."""
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def __call__(self, x: torch.Tensor, class_idx: int | None = None):
        """x: (1,3,H,W). Retorna (mapa 0-1 HxW, classe, probabilidades)."""
        self.model.zero_grad()
        with torch.enable_grad():  # funciona mesmo se chamado dentro de torch.no_grad()
            logits = self.model(x)
            probs = logits.softmax(1)[0].detach().cpu().numpy()
            class_idx = int(logits.argmax(1)) if class_idx is None else class_idx
            logits[0, class_idx].backward()
        weights = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.acts).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx, probs


def overlay(img_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    heat = cv2.resize(heat, (img_rgb.shape[1], img_rgb.shape[0]))
    return cv2.addWeighted(img_rgb, 1 - alpha, heat, alpha, 0)
