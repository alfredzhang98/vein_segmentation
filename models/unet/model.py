""" Full assembly of the parts to form the complete network """

from .parts import *


class UNet(nn.Module):
    """
    base_ch / dropout exist because the vein overfits, and the numbers say it is a
    capacity problem, not a blindness problem:

        Mus-V vein, aggregate Dice    train 0.931   val 0.726
        Mus-V artery                  (no such gap — val 0.90)

    The model sees the vein perfectly on data it has trained on and loses a fifth of it
    on data it has not. It is memorising, and there is plenty of room to: Mus-V's 2203
    training frames come from only 80 probe sweeps — 80 distinct veins — while the
    default width gives 31M parameters. That is 390,000 parameters per vein.

    The artery escapes because it is the same object every time: a round, dark,
    non-collapsible circle. 80 examples span that. A vein does not hold still — it
    flattens under probe pressure, and how it flattens differs by subject — so 80
    examples do not span it, and the spare capacity gets spent remembering them.

    Note what this is NOT: online augmentation was tried first and did not move the val
    vein at all (train loss rose 0.074 -> 0.096, so the memorisation of pixels really was
    broken — the val score simply did not care). Gamma jitter and rotation manufacture
    new views of the same 80 people; they cannot manufacture an 81st.

        base_ch=64 (default)  31.0M params   — the original
        base_ch=32             7.8M
        base_ch=16             1.9M

    dropout applies to the two deepest encoder blocks and the bottleneck, where the
    subject-specific features live; the early layers stay clean because edge and speckle
    filters are shared across every subject and there is nothing to memorise there.
    """

    def __init__(self, n_channels, n_classes, bilinear=False,
                 base_ch=64, dropout=0.0):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.base_ch = base_ch
        self.dropout = dropout

        c = base_ch
        self.inc = (DoubleConv(n_channels, c))
        self.down1 = (Down(c, c * 2))
        self.down2 = (Down(c * 2, c * 4))
        self.down3 = (Down(c * 4, c * 8, dropout=dropout))
        factor = 2 if bilinear else 1
        self.down4 = (Down(c * 8, c * 16 // factor, dropout=dropout))
        self.up1 = (Up(c * 16, c * 8 // factor, bilinear, dropout=dropout))
        self.up2 = (Up(c * 8, c * 4 // factor, bilinear))
        self.up3 = (Up(c * 4, c * 2 // factor, bilinear))
        self.up4 = (Up(c * 2, c, bilinear))
        self.outc = (OutConv(c, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)