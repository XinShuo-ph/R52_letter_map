import torch
import wandb
import tqdm
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import shutil
import glob

# hyperparameters
penalty_power = 1

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--history_length_threshold', type=int, default=1000, help='Number of epochs to track loss history')
parser.add_argument('--steps_without_improvement_threshold', type=int, default=1000, help='Number of epochs without improvement to consider local minimum')
parser.add_argument('--n_epochs', type=int, default=5000000, help='Number of epochs to train for')
parser.add_argument('--checkpoint', type=str, default='checkpoints', help='Directory containing checkpoints')
args = parser.parse_args()

n_epochs = args.n_epochs

# Initialize wandb
wandb.init(project="R52_search", name="letter_map_optimization", settings=wandb.Settings(mode="online"))

# Load the sparse tensor and move to GPU
target_tensor = torch.load("R52_3D_sparse.pt").cuda().to_dense()

# Initialize the letter map randomly on GPU
letter_map = torch.nn.Parameter(1e-2*torch.randn(100, 100, device='cuda', requires_grad=True))

# Define optimizer
optimizer = torch.optim.Adam([letter_map], lr=0.001)

# Create directories if they don't exist
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("known_sols", exist_ok=True)
os.makedirs("known_minima", exist_ok=True)

# Load known solutions and minima
known_solutions = []
known_minima = []

# Load all known solutions
sol_files = sorted(glob.glob('./known_sols/sol_*/letter_map.pt'), 
                  key=lambda x: int(x.split('sol_')[1].split('/')[0]))
for sol_file in sol_files:
    sol = torch.load(sol_file).cuda()
    known_solutions.append(sol)
sol_idx = len(known_solutions)

# Load all known minima
min_files = sorted(glob.glob('./known_minima/min_*/letter_map.pt'),
                  key=lambda x: int(x.split('min_')[1].split('/')[0]))
for min_file in min_files:
    min_point = torch.load(min_file).cuda()
    known_minima.append(min_point)
min_idx = len(known_minima)

print(f"Loaded {len(known_solutions)} known solutions and {len(known_minima)} known minima")

# Load latest checkpoint if available
start_epoch = 0
if os.path.exists(args.checkpoint):
    checkpoints = [f for f in os.listdir(args.checkpoint) if f.endswith('.pt')]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        checkpoint_path = os.path.join(args.checkpoint, latest_checkpoint)
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        letter_map.data = checkpoint['letter_map_state_dict'].cuda()
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")

# Training loop
pbar = tqdm.tqdm(range(start_epoch, start_epoch+n_epochs))

# Create identity matrix on GPU
identity = torch.eye(100, device='cuda')

# Initialize variables for tracking progress
loss_history = []
last_best_loss = float('inf')
steps_without_improvement = 0

for epoch in pbar:
    optimizer.zero_grad()
    
    # Apply letter_map to each dimension
    current = target_tensor
    
    # Contract with first dimension (slot 0)
    current = torch.einsum('ij,iklm->jklm', letter_map, current)
    
    # Contract with second dimension (slot 1) 
    current = torch.einsum('ij,kilm->kjlm', letter_map, current)
    
    # Contract with third dimension (slot 2)
    current = torch.einsum('ij,klim->kljm', letter_map, current)
    
    # Contract with fourth dimension (slot 3)
    current = torch.einsum('ij,klmi->klmj', letter_map, current)
    
    # Permute the dimensions to (4,3,2,1)
    current = current.permute(3,2,1,0)
    
    # Calculate reconstruction loss
    recon_loss = torch.norm(current - target_tensor)
    
    # Calculate identity constraint loss
    identity_loss = torch.norm(letter_map - identity)
    
    # Calculate distance penalty from known solutions and minima
    solution_penalty = torch.tensor(0.0, device='cuda')
    for sol in known_solutions:
        solution_penalty += 1.0 / (torch.norm(letter_map - sol) + 1e-4**(1/penalty_power))**(penalty_power)
    
    minima_penalty = torch.tensor(0.0, device='cuda')
    for min_point in known_minima:
        minima_penalty += 1.0 / (torch.norm(letter_map - min_point) + 1e-4**(1/penalty_power))**(penalty_power)
    
    # Total loss is sum of all terms
    loss = recon_loss + identity_loss + solution_penalty + minima_penalty
    
    # Backward and optimize
    loss.backward()
    optimizer.step()
    
    # Check if we found a new solution
    if recon_loss < 0.5:
        sol_dir = f"./known_sols/sol_{sol_idx}"
        os.makedirs(sol_dir, exist_ok=True)
        torch.save(letter_map.data.cpu(), f"{sol_dir}/letter_map.pt")
        np.savetxt(f"{sol_dir}/letter_map.txt", letter_map.data.cpu().numpy())
        known_solutions.append(letter_map.data.clone())
        sol_idx += 1
        
        # Reinitialize letter_map and optimizer when solution found
        letter_map.data = 1e-2*torch.randn(100, 100, device='cuda')
        optimizer = torch.optim.Adam([letter_map], lr=0.001)
        loss_history = []
        last_best_loss = float('inf')
        steps_without_improvement = 0
    
    # Track progress for detecting local minima
    loss_history.append(loss.item())
    if len(loss_history) > args.history_length_threshold:
        loss_history.pop(0)
        if loss.item() < last_best_loss:
            last_best_loss = loss.item()
        
        if loss.item() < last_best_loss-0.01:
            steps_without_improvement = 0
        else:
            steps_without_improvement += 1
            
        if steps_without_improvement >= args.steps_without_improvement_threshold :
            min_dir = f"./known_minima/min_{min_idx}"
            os.makedirs(min_dir, exist_ok=True)
            torch.save(letter_map.data.cpu(), f"{min_dir}/letter_map.pt")
            np.savetxt(f"{min_dir}/letter_map.txt", letter_map.data.cpu().numpy())
            known_minima.append(letter_map.data.clone())
            min_idx += 1
            
            # Reinitialize letter_map and optimizer when minimum found
            letter_map.data = 1e-2*torch.randn(100, 100, device='cuda')
            optimizer = torch.optim.Adam([letter_map], lr=0.001)
            loss_history = []
            last_best_loss = float('inf')
            steps_without_improvement = 0
    
    # Log basic metrics every epoch
    wandb.log({
        "total_loss": loss.item(),
        "reconstruction_loss": recon_loss.item(),
        "identity_loss": identity_loss.item(),
        "solution_penalty": solution_penalty.item(),
        "minima_penalty": minima_penalty.item(),
        "epoch": epoch,
        "num_solutions": len(known_solutions),
        "num_minima": len(known_minima),
        "steps_without_improvement": steps_without_improvement
    })
    
    # Compute and log detailed statistics every 10 epochs
    if epoch % 10 == 0:
        letter_map_cpu = letter_map.data.cpu().numpy()
        max_val = np.max(letter_map_cpu)
        min_val = np.min(letter_map_cpu)
        mean_val = np.mean(letter_map_cpu)
        var_val = np.var(letter_map_cpu)
        std_val = np.std(letter_map_cpu)
        median_val = np.median(letter_map_cpu)
        skewness = np.mean(((letter_map_cpu - mean_val) / std_val) ** 3)
        kurtosis = np.mean(((letter_map_cpu - mean_val) / std_val) ** 4) - 3
        
        # Create histogram data with 100 bins from -1 to 1
        hist_values, hist_bins = np.histogram(letter_map_cpu.flatten(), bins=100, range=(-1, 1))
        
        # Create histogram data excluding near-zero values
        nonzero_values = letter_map_cpu[np.abs(letter_map_cpu) >= 1e-2]
        hist_values_nz, hist_bins_nz = np.histogram(nonzero_values.flatten(), bins=100, range=(-1, 1))
        
        # Create heatmap with red-white-blue colormap
        plt.figure(figsize=(8, 8))
        plt.imshow(letter_map_cpu, cmap='RdBu', vmin=-1, vmax=1)
        plt.colorbar()
        
        # Save and close the figure to prevent memory leaks
        plt.savefig('temp_heatmap.png')
        plt.close()
        
        # Log detailed statistics to wandb
        wandb.log({
            "letter_map_distribution": wandb.Image('temp_heatmap.png'),
            "letter_map_max": max_val,
            "letter_map_min": min_val,
            "letter_map_mean": mean_val,
            "letter_map_variance": var_val,
            "letter_map_std": std_val,
            "letter_map_median": median_val,
            "letter_map_skewness": skewness,
            "letter_map_kurtosis": kurtosis,
            "letter_map_histogram": wandb.Histogram(
                np_histogram=(hist_values.tolist(), hist_bins.tolist())
            ),
            "letter_map_histogram_nonzero": wandb.Histogram(
                np_histogram=(hist_values_nz.tolist(), hist_bins_nz.tolist())
            ),
            "letter_map_sparsity": np.mean(np.abs(letter_map_cpu) < 1e-3),
        })
        
        # Clean up temporary file
        if os.path.exists('temp_heatmap.png'):
            os.remove('temp_heatmap.png')
    
    # Save checkpoint every 1000 epochs
    if (epoch + 1) % 1000 == 0:
        # Save pytorch checkpoint
        checkpoint = {
            'epoch': epoch,
            'letter_map_state_dict': letter_map.data.cpu(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }
        torch.save(checkpoint, f"checkpoints/letter_map_checkpoint_{epoch+1}.pt")
        
        # Also save as plain text
        letter_map_cpu = letter_map.data.cpu().numpy()
        np.savetxt(f"checkpoints/letter_map_checkpoint_{epoch+1}.txt", letter_map_cpu)
    
    pbar.set_description(f"Loss: {loss.item():.6f}")

# Save the final optimized letter_map
torch.save(letter_map.data.cpu(), "optimized_letter_map.pt")
letter_map_cpu = letter_map.data.cpu().numpy()
np.savetxt("optimized_letter_map.txt", letter_map_cpu)
wandb.finish()
