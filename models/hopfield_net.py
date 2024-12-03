import torch 
from PIL import Image


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
    
    def forward(self, x: torch.Tensor, 
                max_iter: int = 1000, 
                tol: float = 1e-6, 
                mask: torch.Tensor = None): 
        """
        Args: 
            x: Input tensor of shape (batch_size, *pattern_dim)
            max_iter: Maximum number of iterations. 
            tol: Tolerance for the stopping criterion. 
            mask: Binary tensor of shape (batch_size, *pattern_dim) indicating which parts of the state to update.
        """
        # Get original shape for later reshaping
        original_shape = x.shape
        
        # Flatten input
        batch_size = x.shape[0]
        n_features = x.numel() // batch_size
        x = x.reshape(batch_size, n_features)
        
        # Initialize mask if not provided
        if mask is None:
            mask = torch.ones_like(x, dtype=torch.bool)
        else:
            mask = mask.reshape(batch_size, n_features)
        
        for _ in range(max_iter): 
            energy = self.__energy__(x)
            x_new = torch.sign(torch.matmul(x, self.weights))
            
            # Update only the specified subset of the state
            x_new = torch.where(mask, x_new, x)
            
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
    from PIL import Image

    # Read and preprocess the image
    img = Image.open('EquivariantHopfieldNetworks\models\w.png').convert('L')  # convert to grayscale
    img = np.array(img) / 255.0  # normalize to [0,1]
    img = (img > 0.5).astype(float) * 2 - 1  # convert to {-1,1}
    
    # Convert to tensor and add batch dimension
    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
    
    # Create Hopfield Network with this pattern
    net = HopfieldNetwork(img_tensor)

    # Normal image reconstruction
    corrupted_normal = img_tensor.clone()
    w_normal = corrupted_normal.shape[2] // 2
    corrupted_normal[:, :, w_normal:] = -1
    
    mask_normal = torch.zeros_like(corrupted_normal, dtype=torch.bool)
    mask_normal[:, :, w_normal:] = True
    
    recovered_normal = net(corrupted_normal, mask=mask_normal)

    # Rotated image reconstruction
    rotated_img = np.rot90(img)
    rotated_tensor = torch.tensor(rotated_img.copy(), dtype=torch.float32).unsqueeze(0)
    
    corrupted_rotated = rotated_tensor.clone()
    w_rotated = corrupted_rotated.shape[2] // 2
    corrupted_rotated[:, :, w_rotated:] = -1
    
    mask_rotated = torch.zeros_like(corrupted_rotated, dtype=torch.bool)
    mask_rotated[:, :, w_rotated:] = True
    
    recovered_rotated = net(corrupted_rotated, mask=mask_rotated)

    # Plotting
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot normal image reconstruction
    axs[0, 0].imshow(img_tensor[0], cmap="gray")
    axs[0, 0].axis('off')
    axs[0, 0].set_title("Original Pattern")
    axs[0, 1].imshow(corrupted_normal[0], cmap="gray")
    axs[0, 1].axis('off')
    axs[0, 1].set_title("Corrupted Input (Right Half)")
    axs[0, 2].imshow(recovered_normal[0], cmap="gray")
    axs[0, 2].axis('off')
    axs[0, 2].set_title("Recovered Pattern")
    
    # Plot rotated image reconstruction
    axs[1, 0].imshow(img_tensor[0], cmap="gray")
    axs[1, 0].axis('off')
    axs[1, 0].set_title("Original Pattern")
    axs[1, 1].imshow(corrupted_rotated[0], cmap="gray")
    axs[1, 1].axis('off')
    axs[1, 1].set_title("Corrupted Rotated Input (Right Half)")
    axs[1, 2].imshow(recovered_rotated[0], cmap="gray")
    axs[1, 2].axis('off')
    axs[1, 2].set_title("Recovered Pattern")
    
    plt.tight_layout()
    plt.show()
