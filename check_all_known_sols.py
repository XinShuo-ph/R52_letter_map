import numpy as np
import os
import glob

def process_letter_map(letter_map):
    # Divide by 0.706 and round
    processed = np.round(letter_map / 0.706)
    # Convert to int type
    return processed.astype(int)

# Get list of all solution files, including double digit indices
sol_files = sorted(glob.glob('./known_sols/sol_*/letter_map.txt'), 
                  key=lambda x: int(x.split('sol_')[1].split('/')[0]))

if not sol_files:
    print("No solution files found!")
    exit()

# Process all solutions
processed_sols = []
for sol_file in sol_files:
    sol_idx = int(sol_file.split('sol_')[1].split('/')[0])
    letter_map = np.loadtxt(sol_file)
    processed = process_letter_map(letter_map)
    processed_sols.append((sol_idx, processed))
    
    # Save processed solution in Mathematica format
    os.makedirs('./processed_sols', exist_ok=True)
    with open(f'./processed_sols/sol_{sol_idx}.mma', 'w') as f:
        f.write('{\n')  # Start outer array
        for i in range(processed.shape[0]):
            f.write('{')  # Start row
            for j in range(processed.shape[1]):
                if j < processed.shape[1]-1:
                    f.write(f"{processed[i,j]}, ")
                else:
                    f.write(f"{processed[i,j]}")
            if i < processed.shape[0]-1:
                f.write('},\n')  # End row with comma
            else:
                f.write('}\n')  # End last row without comma
        f.write('}')  # End outer array

# Compare each solution with all previous solutions
for i in range(len(processed_sols)):
    curr_idx, curr_sol = processed_sols[i]
    matching_indices = []
    min_diff_count = float('inf')
    min_diff_idx = None
    
    # Compare with all previous solutions
    for j in range(i):
        prev_idx, prev_sol = processed_sols[j]
        if np.array_equal(curr_sol, prev_sol):
            matching_indices.append(prev_idx)
        else:
            # Count number of different elements
            diff_count = np.sum(curr_sol != prev_sol)
            if diff_count < min_diff_count:
                min_diff_count = diff_count
                min_diff_idx = prev_idx
    
    # Print results
    if matching_indices:
        print(f"Solution {curr_idx} is identical to solutions: {matching_indices}")
    else:
        print(f"Solution {curr_idx} differs from all previous solutions")
        print(f"Most similar to solution {min_diff_idx} with {min_diff_count} different elements")
