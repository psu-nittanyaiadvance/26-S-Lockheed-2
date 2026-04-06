import torch
import torch.nn as nn

class DeCURLoss(nn.Module):
    def __init__(self, common_dim, lambda_param=0.0051):
        """
        DeCUR Loss Module. [cite: 130, 519]
        """
        super(DeCURLoss, self).__init__()
        self.common_dim = common_dim
        self.lambda_param = lambda_param

    def forward(self, m1, m1a, m2, m2a):
        # 1. Intra-modal Losses (Target Identity for full vector) [cite: 125, 126]
        l1 = self._loss_c(self._get_cross_correlation(m1, m1a))
        l2 = self._loss_c(self._get_cross_correlation(m2, m2a))

        # 2. Inter-modal Cross Correlation Matrix [cite: 78, 107]
        inter_c = self._get_cross_correlation(m1, m2)

        # 3. Inter-modal Common Loss (Target Identity for top-left) [cite: 110, 112]
        lc = self._loss_c(inter_c[:self.common_dim, :self.common_dim])
        
        # 4. Inter-modal Unique Loss (Target Zero for bottom-right) [cite: 117, 119]
        lu = self._loss_u(inter_c[self.common_dim:, self.common_dim:])

        # Total Loss: L = L_M1 + L_M2 + L_com + L_uni [cite: 131]
        return l1 + l2 + lc + lu

    def off_diagonal(self, x): # Added self for instance method access
        """
        Helper function: Returns a flattened 1D tensor of all off-diagonal 
        elements from a square matrix. 
        """
        n, m = x.shape
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def _loss_c(self, C):
        """
        Loss function for common and intra-modal representations. [cite: 112, 126, 519]
        """
        l_on = torch.diagonal(C).sub(1).pow(2).sum()
        l_off = self.off_diagonal(C).pow(2).sum()
        
        return l_on + (self.lambda_param * l_off)

    def _loss_u(self, C):
        """
        Loss function for unique representations. [cite: 119, 519]
        """
        l_on = torch.diagonal(C).pow(2).sum()
        l_off = self.off_diagonal(C).pow(2).sum()
        
        return l_on + (self.lambda_param * l_off)

    def _get_cross_correlation(self, z1, z2):
        """
        Computes normalized cross-correlation. [cite: 76, 79, 519]
        """
        batch_size = z1.size(0)
        
        # 1. Normalize (biased std matches official implementation) 
        z1_norm = (z1 - z1.mean(0)) / (z1.std(0, unbiased=False) + 1e-8)
        z2_norm = (z2 - z2.mean(0)) / (z2.std(0, unbiased=False) + 1e-8)

        # 2. Matrix Multiplication 
        c = torch.mm(z1_norm.T, z2_norm) / batch_size
        
        return c