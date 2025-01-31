import numpy as np

# Read the letter map from checkpoint
letter_map = np.loadtxt('checkpoints/letter_map_checkpoint_3000.txt')

# Divide by 0.706 and round
letter_map = np.round(letter_map / 0.706)

# Print some basic statistics
print(f"Shape: {letter_map.shape}")
print(f"Min value: {np.min(letter_map)}")
print(f"Max value: {np.max(letter_map)}")
print(f"Mean value: {np.mean(letter_map)}")

# Save the processed map
np.savetxt('processed_letter_map.txt', letter_map)

# change to int type
letter_map = letter_map.astype(int)

# save the matrix in Mathematica format
# Save as regular text file
np.savetxt('processed_letter_map.txt', letter_map, fmt='%d', delimiter=',')

# Save in Mathematica format
with open('processed_letter_map.mma', 'w') as f:
    f.write('{\n')  # Start outer array
    for i in range(letter_map.shape[0]):
        f.write('{')  # Start row
        for j in range(letter_map.shape[1]):
            if j < letter_map.shape[1]-1:
                f.write(f"{letter_map[i,j]}, ")
            else:
                f.write(f"{letter_map[i,j]}")
        if i < letter_map.shape[0]-1:
            f.write('},\n')  # End row with comma
        else:
            f.write('}\n')  # End last row without comma
    f.write('}')  # End outer array

