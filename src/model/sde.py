import torch
import numpy as np

class VPSDE:
    """Base Variance Preserving SDE class"""
    def __init__(self, bmin=0.1, bmax=20.0):
        self.bmin = bmin
        self.bmax = bmax

    def beta(self, t):
        return self.bmin + t * (self.bmax - self.bmin)  # [B] or [B, pred_len]

    def alpha(self, t):
        int_beta = self.bmin * t + 0.5 * (self.bmax - self.bmin) * t ** 2
        return torch.exp(-0.5 * int_beta)  # [B, pred_len]

    def f(self, x, t):
        beta_t = self.beta(t).view(*t.shape, 1, 1)  # [B, pred_len, 1, 1]
        return -0.5 * beta_t * x

    def g(self, t):
        return torch.sqrt(self.beta(t)).view(*t.shape, 1, 1)  # [B, pred_len, 1, 1]

    def p(self, x, t):
        """
        Forward process.
        
        Args:
            x: Input tensor [B, T, L, D] or [B, T, D]
            t: Time parameter [B] or [B, T]
            
        Returns:
            mu: Mean of forward process
            std: Standard deviation
        """
        # Add level dimension if needed
        if x.dim() == 3:
            x = x.unsqueeze(2)  # [B, T, 1, D]
            
        # Ensure t has time dimension
        if t.dim() == 1:
            t = t.unsqueeze(1)  # [B, 1]
            
        alpha_t = self.alpha(t).view(*t.shape, 1, 1)  # [B, T, 1, 1]
        mu = alpha_t * x
        std = torch.sqrt(torch.clamp(1.0 - alpha_t ** 2, min=1e-5))
        return mu, std

class CovariantVPSDE(VPSDE):
    """
    VPSDE with covariant noise that respects relationships between wavelet levels.
    
    The noise correlation structure is based on the intuition that:
    1. Adjacent wavelet levels should have stronger correlations
    2. The approximation coefficients (level 0) influence all other levels
    3. Higher frequency levels have weaker correlations with low frequency levels
    
    Args:
        bmin: Minimum noise level
        bmax: Maximum noise level
        level_correlations: Optional custom correlation matrix [L+1, L+1]
        num_features: Number of features in the data
    """
    def __init__(self, bmin=0.1, bmax=20.0, level_correlations=None, num_features=4):
        super().__init__(bmin, bmax)
        self.num_features = num_features
        
        # Default correlations between wavelet levels if not provided
        if level_correlations is None:
            # Create correlation matrix with exponential decay
            size = 4  # L+1 for L=3
            distances = np.abs(np.arange(size)[:, None] - np.arange(size))
            correlations = np.exp(-distances * 0.5)  # Exponential decay with distance
            # Increase correlation with approximation coefficients (level 0)
            correlations[0] += 0.2
            correlations[:,0] += 0.2
            # Normalize to valid correlation matrix
            correlations = np.clip(correlations, 0, 1)
            correlations = (correlations + correlations.T) / 2  # Ensure symmetry
            level_correlations = torch.from_numpy(correlations.astype(np.float32))
        
        self.register_buffer('level_correlations', level_correlations)
        # Compute Cholesky decomposition once
        self.register_buffer('cholesky_L', torch.linalg.cholesky(level_correlations))
    
    def register_buffer(self, name, tensor):
        """Helper to register a buffer that works outside nn.Module"""
        if not hasattr(self, '_buffers'):
            self._buffers = {}
        self._buffers[name] = tensor.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    def g(self, t):
        """
        Generate correlated noise across wavelet levels.
        
        Args:
            t: Time parameter [B] or [B, pred_len]
            
        Returns:
            Noise with shape [B, pred_len, L+1, D] where:
            - B is batch size
            - pred_len is prediction length
            - L+1 is number of wavelet levels
            - D is number of features
        """
        base_std = super().g(t)  # [B, pred_len, 1]
        B, T, _ = base_std.shape
        L = self.level_correlations.shape[0]
        
        # Generate base noise
        noise = torch.randn(B, T, L, self.num_features, 
                          device=base_std.device)  # [B, T, L, D]
        
        # Apply correlation structure across levels
        # Reshape for matrix multiplication
        noise_flat = noise.view(-1, L, self.num_features)  # [B*T, L, D]
        # Apply Cholesky to correlate levels
        correlated = torch.matmul(self.cholesky_L, 
                                noise_flat.transpose(-2, -1))  # [B*T, L, D]
        # Reshape back
        correlated = correlated.transpose(-2, -1).view(B, T, L, self.num_features)
        
        # Scale by base standard deviation
        return base_std.unsqueeze(-1) * correlated  # [B, T, L, D]
    
    def p(self, x, t):
        """
        Forward process with correlated noise.
        
        Args:
            x: Input tensor [B, T, L, D]
            t: Time parameter [B] or [B, T]
            
        Returns:
            mu: Mean of the forward process
            std: Standard deviation with correlation structure
        """
        alpha_t = self.alpha(t).view(t.shape[0], -1, 1, 1)  # [B, T, 1, 1]
        mu = alpha_t * x
        std = torch.sqrt(torch.clamp(1.0 - alpha_t ** 2, min=1e-5))
        return mu, std
