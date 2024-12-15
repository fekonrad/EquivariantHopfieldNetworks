# Equivariant Hopfield Networks

![Energy Function](/energy_fct_C4.png)

# Examples 
## Euclidean Plane $\mathbb R^2$ and $90°$ Rotations $C_4$
A Hopfield Network with rotationaly symmetry is implemented in `models/rotation_hn.py`. 
Given any set of patterns, it can store not only the patterns themselves, but also their rotated versions, without any data augmentation. 
```python
# Initialize the Hopfield Network with 1 Pattern: 
patterns = torch.tensor([[1.0, 0.0]]) 
rotation_hn = RotationHopfieldNetwork2D(patterns)
```
The network then correctly retrieves the rotated version of the pattern that is "most similar" to any given input $x$:
```python
x = torch.tensor([[2.0, 0.0], 
                    [0.0, 0.5], 
                    [-1.3, 0.5], 
                    [1.0, -2.0]])
print(f"Output (rounded): \n {rotation_hn(x).detach().clone().round().numpy()}")
```
This will return: 
```
Output (rounded): 
 [[ 1.  0.]
 [ 0.  1.]
 [ -1.  0.]
 [ 0. -1.]]
```
