"""Shared mixed-supervision objective, unchanged from the v11 U-Net trainer.

full3: background / vein / artery; artery: p2 vs p0+p1;
vessel: p1+p2 vs p0. Empty reviewed masks remain valid supervision.
"""
from typing import Dict, Any, List
import torch
import torch.nn.functional as F
from data.pipeline.dataPrepare import MODE_IDS, CLASS_BG, CLASS_VEIN, CLASS_ARTERY, CLASS_VESSEL

LABEL_MODES = ("full3", "vessel", "artery")


def _soft_dice(prob: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Soft Dice on a single foreground probability map. prob/target: (N, H, W)."""
    inter    = 2 * (prob * target).sum(dim=(-1, -2))
    sets_sum = prob.sum(dim=(-1, -2)) + target.sum(dim=(-1, -2))
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)
    return ((inter + eps) / (sets_sum + eps)).mean()


def _soft_tversky(prob: torch.Tensor, target: torch.Tensor,
                  alpha: float, beta: float, eps: float = 1e-6) -> torch.Tensor:
    """
    Tversky index. alpha weights false positives, beta weights false negatives;
    alpha = beta = 0.5 reduces exactly to Dice.

    Why this exists: measured on Mus-V val, 37% of true VEIN pixels were being called
    background, while only 1.3% were called artery — the model was not confusing the
    two vessels, it was missing the vein outright. A vein is thin, compressed against
    the tissue and low-contrast, so under a loss that prices a false positive and a
    false negative identically (Dice does), the safe play is to predict background.
    Setting beta > alpha makes missing a vein pixel more expensive than inventing one,
    which is the asymmetry the problem actually has.
    """
    tp = (prob * target).sum(dim=(-1, -2))
    fp = (prob * (1 - target)).sum(dim=(-1, -2))
    fn = ((1 - prob) * target).sum(dim=(-1, -2))
    return ((tp + eps) / (tp + alpha * fp + beta * fn + eps)).mean()


def _region_loss(prob: torch.Tensor, target: torch.Tensor,
                 cfg: Dict[str, Any]) -> torch.Tensor:
    """1 - Dice, or 1 - Tversky when TVERSKY_BETA is set."""
    beta = cfg.get("tversky_beta")
    if not beta:
        return 1 - _soft_dice(prob, target)
    return 1 - _soft_tversky(prob, target, cfg.get("tversky_alpha", 1.0 - beta), beta)


def partial_label_loss(logits: torch.Tensor, target: torch.Tensor,
                       modes: List[str],
                       class_weights: torch.Tensor = None,
                       cfg: Dict[str, Any] = None) -> torch.Tensor:
    """
    logits (B, 3, H, W) — raw. target (B, 1, H, W) long, ids in that sample's own
    label space. modes: per-sample label_mode, len B.
    """
    cfg    = cfg or {}
    if len(modes) != logits.shape[0] or not modes or set(modes) - set(LABEL_MODES):
        raise ValueError("Each sample must have a supported label_mode")
    logits = logits.float()
    logp   = F.log_softmax(logits, dim=1)          # (B, 3, H, W)
    tgt    = target.squeeze(1).long()              # (B, H, W)

    losses = []
    for mode in LABEL_MODES:
        sel = torch.tensor([m == mode for m in modes], device=logits.device)
        if not sel.any():
            continue
        lp, y = logp[sel], tgt[sel]

        # The ids ARE the meaning now (dataPrepare: 0 bg / 1 vein / 2 artery /
        # 3 vessel-untyped). A sample routed to the wrong marginal would otherwise
        # supervise the wrong channel and merely look like a bad epoch.
        assert set(torch.unique(y).tolist()) <= MODE_IDS[mode], (
            f"label_mode='{mode}' 收到了 id {sorted(torch.unique(y).tolist())}，"
            f"只允许 {sorted(MODE_IDS[mode])}"
        )

        if mode == "full3":
            ce = F.nll_loss(lp, y, weight=class_weights)
            # Per-class region loss rather than dice_coeff's 3-class average: that
            # average is dominated by the background channel (96% of pixels, always
            # near-perfect), which drowns out the vein term we are actually trying to
            # move. Averaging the two vessel classes only puts the pressure where the
            # error is.
            p = lp.exp()
            region = torch.stack([
                _region_loss(p[:, c], (y == c).float(), cfg)
                for c in (CLASS_VEIN, CLASS_ARTERY)
            ]).mean()

        elif mode == "vessel":
            # marginal over {bg} vs {vein, artery} — the split between them stays free
            log_fg = torch.logsumexp(lp[:, [CLASS_VEIN, CLASS_ARTERY]], dim=1)  # log(p1+p2)
            log_bg = lp[:, CLASS_BG]                        # log(p0) == log(1 - p_fg)
            yf     = (y == CLASS_VESSEL).float()
            ce     = -(yf * log_fg + (1 - yf) * log_bg).mean()
            region = _region_loss(log_fg.exp(), yf, cfg)

        elif mode == "artery":
            # marginal over {bg, vein} vs {artery}; the bg/vein split stays free, so an
            # unlabelled jugular vein predicted here costs exactly nothing
            log_fg = lp[:, CLASS_ARTERY]                                        # log(p2)
            log_bg = torch.logsumexp(lp[:, [CLASS_BG, CLASS_VEIN]], dim=1)      # log(p0+p1)
            yf     = (y == CLASS_ARTERY).float()
            ce     = -(yf * log_fg + (1 - yf) * log_bg).mean()
            region = _region_loss(log_fg.exp(), yf, cfg)

        losses.append((ce + region) * sel.sum())

    return torch.stack(losses).sum() / len(modes)
