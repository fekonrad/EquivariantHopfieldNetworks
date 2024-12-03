import torch 


class RotationHopfieldNetwork2D(torch.nn.Module):
    """
        A Hopfield Network designed for 2D data, whose energy function
        is invariant with respect to rotations (90°, 180°, 270°).
        As a consequence, it can store all rotated versions of a given pattern, 
        without it needing to see each rotated version individually! 
        We do so by augmenting the input with a feature map that is invariant with respect to rotations. 
    """
    def __init__(self, patterns: torch.Tensor): 
        """
            Args: 
                num_patterns: Number of patterns to learn. 
        """
        super().__init__()
        self.patterns = patterns

    def __inv_feature_map__(self, x: torch.Tensor): 
        # Ensure x is of shape (b, 2)
        assert x.shape[1] == 2, "Input tensor must have shape (b, 2)"
        
        # Calculate the features
        feature1 = x[:, 0]**2 + x[:, 1]**2
        feature2 = x[:, 0]**2 * x[:, 1]**2
        feature3 = x[:, 0]**3 * x[:, 1] - x[:, 0] * x[:, 1]**3
        
        # Stack the features into a tensor of shape (b, 3)
        return torch.stack((feature1, feature2, feature3), dim=1)
    
    
    def __energy__(self, x: torch.Tensor, beta: float = 1.0): 
        """
        Compute energy E(x) = -lse(beta, phi(p)^T phi(x)) + |x|^2/2
        
        Args:
            x: Input tensor of shape (b, 2)
            beta: Temperature parameter for log-sum-exp
        """
        # Compute feature map phi(x)
        phi_x = self.__inv_feature_map__(x)  # shape: (b, 3)
        
        # Compute phi(p)^T phi(x) for all patterns
        phi_patterns = self.__inv_feature_map__(self.patterns)  # shape: (n_patterns, 3)
        logits = torch.matmul(phi_x, phi_patterns.T)  # shape: (b, n_patterns)
        
        # Compute log-sum-exp carefully to avoid overflow
        # lse(beta, x) = (1/beta) * log(sum(exp(beta * x)))
        max_logits = torch.max(beta * logits, dim=1, keepdim=True)[0]
        exp_term = torch.exp(beta * logits - max_logits)
        lse = (1.0 / beta) * (max_logits + torch.log(torch.sum(exp_term, dim=1)))
        
        # Compute L2 regularization term |phi(x)|^2/2
        l2_term = 0.5 * torch.sum(phi_x**2, dim=1)
        
        # Return negative lse plus L2 term
        return -lse + l2_term

    def forward(self, x: torch.Tensor, max_iter: int = 1000, tol: float = 1e-6, mask: torch.Tensor = None): 
        # Initialize mask if not provided - In this case no part of input is masked.
        if mask is None:
            mask = torch.ones_like(x, dtype=torch.bool)
        
        x = x.detach().clone()
        x.requires_grad = True
        
        for _ in range(max_iter): 
            # Compute energy
            energy = self.__energy__(x, beta=0.01)
            energy.sum().backward()
            
            # Gradient descent step
            with torch.no_grad():
                x_new = x - 0.01 * x.grad
                
                # Update only the specified subset of the state
                x_new = torch.where(mask, x_new, x)
                
                # Check convergence
                if torch.all(torch.abs(x_new - x) < tol):
                    break
                    
                # Update x and reset gradients
                x = x_new.detach().clone()
                x.requires_grad = True
                
        return x

if __name__ == "__main__": 
    """
        Here we only give the network the pattern (1, 0) and see 
        if it can retrieve the rotated versions (0, 1), (-1, 0), and (0, -1).
    """
    patterns = torch.tensor([[1.0, 0.0]]) 
    rotation_hn = RotationHopfieldNetwork2D(patterns)

    x = torch.tensor([[2.0, 0.0], 
                      [0.0, 0.5], 
                      [-1.3, 0.5], 
                      [1.0, -2.0]])
    print(f"Inputs: \n {x.numpy()}")
    print(f"Output (rounded): \n {rotation_hn(x).detach().clone().round().numpy()}")

    # Problem: Seems to diverge! 
    # But at least the "direction" of the diverging states is correct! 
    # And equivariance of the updates does seem to hold!

