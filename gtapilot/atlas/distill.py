from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TeacherToStudentProjector(nn.Module):
    def __init__(self, teacher_dim: int = 640, student_dim: int = 448):
        super().__init__()
        self.proj = nn.Linear(teacher_dim, student_dim)

    def forward(self, teacher_tokens: torch.Tensor) -> torch.Tensor:
        return self.proj(teacher_tokens)


def distill_logits(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature**2)


def distill_features(student_tokens: torch.Tensor, teacher_tokens: torch.Tensor, projector: TeacherToStudentProjector | None = None) -> torch.Tensor:
    if projector is not None:
        teacher_tokens = projector(teacher_tokens)
    return F.smooth_l1_loss(student_tokens, teacher_tokens)
