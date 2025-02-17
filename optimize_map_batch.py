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
steps_without_improvement_threshold = 1000
num_parallel = 32  # Number of parallel optimization attempts

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default='checkpoints', help='Directory containing checkpoints')
args = parser.parse_args()

# Initialize wandb
wandb.init(project="R52_search", name="letter_map_optimization", settings=wandb.Settings(mode="online"))

# Load the sparse tensor and move to GPU
target_tensor = torch.load("R52_3D_sparse.pt").cuda().to_dense()

# Initialize multiple letter maps randomly on GPU
letter_maps = torch.nn.Parameter(1e-2*torch.randn(num_parallel, 100, 100, device='cuda', requires_grad=True))

# Define optimizer
optimizer = torch.optim.Adam([letter_maps], lr=0.001)

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
        letter_maps.data = checkpoint['letter_maps_state_dict'].cuda()
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")

# Training loop
n_epochs = 5000000
pbar = tqdm.tqdm(range(start_epoch, n_epochs))

# Create identity matrix on GPU
identity = torch.eye(100, device='cuda')

# Initialize variables for tracking progress
loss_history = []
last_best_loss = float('inf')
steps_without_improvement = 0

for epoch in pbar:
    optimizer.zero_grad()
    
    # Process all attempts in parallel
    current = target_tensor.unsqueeze(0).expand(num_parallel, -1, -1, -1, -1)
    
    # Contract with all dimensions in parallel using batched einsum
    current = torch.einsum('bij,biklm->bjklm', letter_maps, current)
    current = torch.einsum('bij,bkilm->bkjlm', letter_maps, current)
    current = torch.einsum('bij,bklim->bkljm', letter_maps, current)
    current = torch.einsum('bij,bklmi->bklmj', letter_maps, current)
        
    current = current.permute(0, 4, 3, 2, 1)
    
    # Calculate losses for all attempts using linalg.norm instead of norm
    recon_losses = torch.linalg.vector_norm(current - target_tensor.unsqueeze(0), dim=(1,2,3,4))
    identity_losses = torch.linalg.vector_norm(letter_maps - identity.unsqueeze(0), dim=(1,2))
    
    # Calculate penalties
    solution_penalties = torch.zeros(num_parallel, device='cuda')
    minima_penalties = torch.zeros(num_parallel, device='cuda')
    
    for sol in known_solutions:
        solution_penalties += 1.0 / (torch.linalg.vector_norm(letter_maps - sol.unsqueeze(0), dim=(1,2)) + 1e-4**(1/penalty_power))**(penalty_power)
    
    for min_point in known_minima:
        minima_penalties += 1.0 / (torch.linalg.vector_norm(letter_maps - min_point.unsqueeze(0), dim=(1,2)) + 1e-4**(1/penalty_power))**(penalty_power)

    # Total loss is sum of all terms
    losses = recon_losses + identity_losses + solution_penalties + minima_penalties
    loss = losses.mean()
    
    # Backward and optimize
    loss.backward()
    optimizer.step()
    
    # Find best performing letter map for solutions (using recon_losses)
    best_recon_idx = torch.argmin(recon_losses)
    min_recon_loss = recon_losses[best_recon_idx]
    best_recon_map = letter_maps.data[best_recon_idx]
    
    # Find best performing letter map for minima (using total losses)
    best_total_idx = torch.argmin(losses)
    min_total_loss = losses[best_total_idx]
    best_total_map = letter_maps.data[best_total_idx]
    
    # Check if we found a new solution (using recon_losses criterion)
    if min_recon_loss < 0.5:
        sol_dir = f"./known_sols/sol_{sol_idx}"
        os.makedirs(sol_dir, exist_ok=True)
        torch.save(best_recon_map.cpu(), f"{sol_dir}/letter_map.pt")
        np.savetxt(f"{sol_dir}/letter_map.txt", best_recon_map.cpu().numpy())
        known_solutions.append(best_recon_map.clone())
        sol_idx += 1
        
        # Reinitialize letter_maps and optimizer when solution found
        letter_maps.data = 1e-2*torch.randn(num_parallel, 100, 100, device='cuda')
        optimizer = torch.optim.Adam([letter_maps], lr=0.001)
        loss_history = []
        last_best_loss = float('inf')
        steps_without_improvement = 0
    
    # Track progress for detecting local minima (using total losses criterion)
    loss_history.append(min_total_loss.item())
    if len(loss_history) > steps_without_improvement_threshold:
        loss_history.pop(0)
        if min_total_loss.item() < last_best_loss:
            last_best_loss = min_total_loss.item()
            steps_without_improvement = 0
        else:
            steps_without_improvement += 1
            
        if steps_without_improvement >= steps_without_improvement_threshold and max(loss_history) - min(loss_history) < 0.01:
            min_dir = f"./known_minima/min_{min_idx}"
            os.makedirs(min_dir, exist_ok=True)
            torch.save(best_total_map.cpu(), f"{min_dir}/letter_map.pt")
            np.savetxt(f"{min_dir}/letter_map.txt", best_total_map.cpu().numpy())
            known_minima.append(best_total_map.clone())
            min_idx += 1
            
            # Reinitialize letter_maps and optimizer when minimum found
            letter_maps.data = 1e-2*torch.randn(num_parallel, 100, 100, device='cuda')
            optimizer = torch.optim.Adam([letter_maps], lr=0.001)
            loss_history = []
            last_best_loss = float('inf')
            steps_without_improvement = 0
    
    # Log basic metrics every epoch
    wandb.log({
        "total_loss": loss.item(),
        "min_reconstruction_loss": min_recon_loss.item(),
        "identity_loss": identity_losses[best_total_idx].item(),
        "solution_penalty": solution_penalties[best_total_idx].item(),
        "minima_penalty": minima_penalties[best_total_idx].item(),
        "epoch": epoch,
        "num_solutions": len(known_solutions),
        "num_minima": len(known_minima)
    })
    
    # Compute and log detailed statistics every 10 epochs
    if epoch % 10 == 0:
        best_letter_map_cpu = best_total_map.cpu().numpy()
        max_val = np.max(best_letter_map_cpu)
        min_val = np.min(best_letter_map_cpu)
        mean_val = np.mean(best_letter_map_cpu)
        var_val = np.var(best_letter_map_cpu)
        std_val = np.std(best_letter_map_cpu)
        median_val = np.median(best_letter_map_cpu)
        skewness = np.mean(((best_letter_map_cpu - mean_val) / std_val) ** 3)
        kurtosis = np.mean(((best_letter_map_cpu - mean_val) / std_val) ** 4) - 3
        
        # Create histogram data with 100 bins from -1 to 1
        hist_values, hist_bins = np.histogram(best_letter_map_cpu.flatten(), bins=100, range=(-1, 1))
        
        # Create histogram data excluding near-zero values
        nonzero_values = best_letter_map_cpu[np.abs(best_letter_map_cpu) >= 1e-2]
        hist_values_nz, hist_bins_nz = np.histogram(nonzero_values.flatten(), bins=100, range=(-1, 1))
        
        # Create heatmap with red-white-blue colormap
        plt.figure(figsize=(8, 8))
        plt.imshow(best_letter_map_cpu, cmap='RdBu', vmin=-1, vmax=1)
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
            "letter_map_sparsity": np.mean(np.abs(best_letter_map_cpu) < 1e-3),
        })
        
        # Clean up temporary file
        if os.path.exists('temp_heatmap.png'):
            os.remove('temp_heatmap.png')
    
    # Save checkpoint every 1000 epochs
    if (epoch + 1) % 1000 == 0:
        # Save pytorch checkpoint
        checkpoint = {
            'epoch': epoch,
            'letter_maps_state_dict': letter_maps.data.cpu(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }
        torch.save(checkpoint, f"checkpoints/letter_map_checkpoint_{epoch+1}.pt")
        
        # Also save best letter map as plain text
        np.savetxt(f"checkpoints/letter_map_checkpoint_{epoch+1}.txt", best_total_map.cpu().numpy())
    
    pbar.set_description(f"Loss: {loss.item():.6f}, Min Recon Loss: {min_recon_loss.item():.6f}")

# Save the final optimized letter_map (best performing one)
torch.save(best_total_map.cpu(), "optimized_letter_map.pt")
np.savetxt("optimized_letter_map.txt", best_total_map.cpu().numpy())
wandb.finish()