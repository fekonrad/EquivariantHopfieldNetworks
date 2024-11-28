import torch 


class HopfieldNetwork(torch.nn.Module): 
    """
        A standard Hopfield network. 
    """
    def __init__(self, patterns: torch.Tensor): 
        super().__init__()
        self.weights = self._init_weights(patterns)

    def _init_weights(self, patterns: torch.Tensor):
        # Get batch size and total number of features
        batch_size = patterns.shape[0]
        n_features = patterns.numel() // batch_size
        
        # Reshape patterns to (batch_size, n_features)
        patterns_flat = patterns.reshape(batch_size, n_features)
        
        # Initialize weights matrix
        weights = torch.zeros((n_features, n_features))
        
        # Sum outer products for all patterns
        for pattern in patterns_flat:
            weights += torch.outer(pattern, pattern)
        
        # Zero out diagonal elements
        weights = weights.masked_fill(torch.eye(n_features).bool(), 0)
        
        return weights
    
    def __energy__(self, x: torch.Tensor): 
        # Energy = -0.5 * x^T * W * x
        # For batched input: calculate energy for each sample
        x_W = torch.matmul(x, self.weights)  # (batch_size, n_features)
        return -0.5 * torch.sum(x * x_W, dim=1)  # sum over features
    
    def forward(self, x: torch.Tensor, max_iter: int = 1000, tol: float = 1e-6): 
        """
        Args: 
            x: Input tensor of shape (batch_size, *pattern_dim)
            max_iter: Maximum number of iterations. 
            tol: Tolerance for the stopping criterion. 
        """
        # Get original shape for later reshaping
        original_shape = x.shape
        
        # Flatten input
        batch_size = x.shape[0]
        n_features = x.numel() // batch_size
        x = x.reshape(batch_size, n_features)
        
        for _ in range(max_iter): 

            energy = self.__energy__(x)
            x_new = torch.sign(torch.matmul(x, self.weights))
            new_energy = self.__energy__(x_new)
        
            # Check if all samples in batch have converged
            if torch.all(torch.abs(energy - new_energy) < tol):
                break
            
            x = x_new
        
        # Reshape back to original dimensions
        x = x.reshape(original_shape)
        return x
        

if __name__ == "__main__": 
    import numpy as np 
    import matplotlib.pyplot as plt 

    size = (32, 32)
    tile_size = 2
    img = np.zeros(size)
    for i in range(0, size[0], tile_size * 2):
        for j in range(0, size[1], tile_size * 2):
            img[i:i+tile_size, j:j+tile_size] = 1
            img[i+tile_size:i+tile_size*2, j+tile_size:j+tile_size*2] = 1

    img = torch.tensor(img, dtype=torch.float32).view(1, -1)
    net = HopfieldNetwork(img)

    x = (2 * torch.randint(0, 2, (5, size[0], size[1])) - 1).float()
    y = net(x)

    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(img.view(size), cmap="gray")
    axs[0].set_title("Original")
    axs[1].imshow(x[0].view(size), cmap="gray")
    axs[1].set_title("Input")
    axs[2].imshow(y[0].view(size), cmap="gray")
    axs[2].set_title("Recovered")
    plt.show()
