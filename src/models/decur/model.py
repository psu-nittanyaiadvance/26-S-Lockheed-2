
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UNetEncoder(nn.Module):
    """
    The encoder (downsampling) half of a U-Net, repurposed as a standalone
    feature extractor for self-supervised learning.

    This is your existing U-Net encoder path:
        inc:   input → 64 channels
        down1: 64 → 128
        down2: 128 → 256
        down3: 256 → 512
        down4: 512 → 1024

    We add global average pooling at the end to collapse the spatial dimensions
    (H x W) into a single 1024-dim feature vector, similar to how ResNet-50
    uses avgpool to produce a 2048-dim vector.

    The decoder (upsampling) path is NOT included here — it's not needed for
    pretraining. DeCUR only needs a feature vector, not a segmentation map.
    After pretraining, you can plug these trained encoder weights back into
    your full U-Net for downstream segmentation.

    Args:
        n_channels: Number of input channels (2 for SAR, 13 for optical)
    """

    def __init__(self, n_channels):
        super().__init__()

        # Encoder path — same as your U-Net's downsampling half
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        # Global average pooling: collapses (batch, 1024, H, W) → (batch, 1024)
        # This is what converts spatial feature maps into a single feature vector
        # that can be fed into the projector MLP.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # Encoder path
        x = self.inc(x)       # (batch, 64, H, W)
        x = self.down1(x)     # (batch, 128, H/2, W/2)
        x = self.down2(x)     # (batch, 256, H/4, W/4)
        x = self.down3(x)     # (batch, 512, H/8, W/8)
        x = self.down4(x)     # (batch, 1024, H/16, W/16)

        # Global average pooling → flatten to feature vector
        x = self.avgpool(x)   # (batch, 1024, 1, 1)
        x = torch.flatten(x, 1)  # (batch, 1024)

        return x


class DeCUR(nn.Module):
    """
    DeCUR: Decoupling Common and Unique Representations.
    Uses two U-Net encoders (one for SAR, one for optical) trained with:
        1. Intra-modal Barlow Twins loss (per modality, all dimensions)
        2. Cross-modal common loss (common dimensions → identity correlation)
        3. Cross-modal unique loss (unique dimensions → zero correlation)
    Args:
        args: Namespace with the following required attributes:
            - dim_common (int): Number of embedding dimensions designated as "common"
            - lambd (float): Trade-off weight for off-diagonal loss terms
            - batch_size (int): Batch size per GPU (used for correlation normalization)
    """

    def __init__(self, args):
        super().__init__()
        self.args = args

        # Encoders — one per modality
        self.encoder_SAR = UNetEncoder(n_channels=2)
        self.encoder_OPT = UNetEncoder(n_channels=13)

        # Projectors — independent weights per modality
        sizes = [1024, 8192, 8192, 8192]
        self.projector_SAR = self._build_projector(sizes)
        self.projector_OPT = self._build_projector(sizes)

        # Batch norm for standardizing embeddings before correlation
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def _build_projector(self, sizes):
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        return nn.Sequential(*layers)
    
    def bt_loss_cross(self, z1, z2):
        """
        Cross-modal decoupled loss — the core of DeCUR.

        Splits the correlation matrix into common and unique submatrices:
            - Common dims → target is identity (correlate across modalities)
            - Unique dims → target is zero (decorrelate across modalities)

        Args:
            z1: SAR embeddings, shape (batch_size, 8192)
            z2: Optical embeddings, shape (batch_size, 8192)

        Returns:
            Tuple of (loss_c, on_diag_c, off_diag_c, loss_u, on_diag_u, off_diag_u)
        """
        # Cross-correlation matrix between SAR and optical embeddings
        c = self.bn(z1).T @ self.bn(z2)

        # Normalize across all GPUs
        c.div_(self.args.batch_size * 4)
        torch.distributed.all_reduce(c)

        # Split into common and unique submatrices
        dim_c = self.args.dim_common
        c_c = c[:dim_c, :dim_c]   # common submatrix
        c_u = c[dim_c:, dim_c:]   # unique submatrix

        # COMMON LOSS: drive toward identity (C_ii → 1, C_ij → 0)
        on_diag_c = torch.diagonal(c_c).add_(-1).pow_(2).sum()
        off_diag_c = off_diagonal(c_c).pow_(2).sum()
        loss_c = on_diag_c + self.args.lambd * off_diag_c

        # UNIQUE LOSS: drive toward zero (C_ii → 0, C_ij → 0)
        on_diag_u = torch.diagonal(c_u).pow_(2).sum()
        off_diag_u = off_diagonal(c_u).pow_(2).sum()
        loss_u = on_diag_u + self.args.lambd * off_diag_u

        return loss_c, on_diag_c, off_diag_c, loss_u, on_diag_u, off_diag_u

    def bt_loss_single(self, z1, z2):
        """
        Intra-modal Barlow Twins loss for a single modality.

        Standard Barlow Twins on ALL dimensions (common + unique).
        Prevents unique dimensions from collapsing by forcing them to
        encode meaningful, augmentation-invariant information.

        Args:
            z1: Embeddings from augmented view 1, shape (batch_size, 8192)
            z2: Embeddings from augmented view 2, shape (batch_size, 8192)

        Returns:
            Tuple of (loss, on_diag, off_diag)
        """
        c = self.bn(z1).T @ self.bn(z2)

        c.div_(self.args.batch_size * 4)
        torch.distributed.all_reduce(c)

        # Target: identity (all dims invariant to augmentation, all decorrelated)
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.args.lambd * off_diag

        return loss, on_diag, off_diag

    def forward(self, y1_1, y1_2, y2_1, y2_2):
        """
        Full DeCUR forward pass.

        Args:
            y1_1: SAR augmented view 1 — shape (batch, 2, H, W)
            y1_2: SAR augmented view 2 — shape (batch, 2, H, W)
            y2_1: Optical augmented view 1 — shape (batch, 13, H, W)
            y2_2: Optical augmented view 2 — shape (batch, 13, H, W)

        Returns:
            Embeddings
    
        """
        # 
        # STEP 1: Encode — each modality through its own encoder
        # 
        f1_1 = self.encoder_SAR(y1_1)   # SAR view 1 → (batch, 1024)
        f1_2 = self.encoder_SAR(y1_2)   # SAR view 2 → (batch, 1024)
        f2_1 = self.encoder_OPT(y2_1)   # Optical view 1 → (batch, 1024)
        f2_2 = self.encoder_OPT(y2_2)   # Optical view 2 → (batch, 1024)

        
        # STEP 2: Project — into 8192-dim embedding space
        
        z1_1 = self.projector_SAR(f1_1)  # (batch, 8192)
        z1_2 = self.projector_SAR(f1_2)  # (batch, 8192)
        z2_1 = self.projector_OPT(f2_1)  # (batch, 8192)
        z2_2 = self.projector_OPT(f2_2)  # (batch, 8192)

        
    

        return z1_1, z1_2, z2_1, z2_2