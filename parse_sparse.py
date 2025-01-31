import torch
import re

def parse_mathematica_sparse(filename):
    # Read the file
    with open(filename, 'r') as f:
        content = f.read()
    
    # Extract coordinates and values using regex
    # Pattern matches {i, j, k, l} -> value
    pattern = r'\{(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\}\s*->\s*([+-]?\d*/?\.?\d*)'
    matches = re.findall(pattern, content)
    
    # Convert to lists for PyTorch
    indices = []
    values = []
    
    for match in matches:
        # Convert coordinates to 0-based indexing
        i, j, k, l = [int(x)-1 for x in match[:4]]  
        
        # Handle fractional values
        value_str = match[4]
        if '/' in value_str:
            num, denom = value_str.split('/')
            value = float(num) / float(denom)
        else:
            value = float(value_str)
            
        indices.append([i, j, k, l])
        values.append(value)
    
    # Convert to PyTorch tensors
    indices = torch.tensor(indices).t()  # Convert to 2D tensor and transpose
    values = torch.tensor(values)
    
    # Create sparse tensor
    sparse_tensor = torch.sparse_coo_tensor(
        indices, 
        values,
        size=(100, 100, 100, 100),
        dtype=torch.float32
    )
    
    return sparse_tensor

# Usage
filename = "R52_3D_sparsemap.m"
sparse_tensor = parse_mathematica_sparse(filename)
print(f"Sparse tensor shape: {sparse_tensor.size()}")
print(f"Number of non-zero elements: {sparse_tensor._nnz()}")

# Save the sparse tensor
torch.save(sparse_tensor, "R52_3D_sparse.pt")
