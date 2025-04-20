import numpy as np
import matplotlib.pyplot as plt

# Set grid dimensions based on the header information from the .dat file.
# According to the header, Nxh = 1024, Nyh = 1024.
Nxh, Nyh = 514, 1026 

# Load the data from the .dat file, skipping header lines that start with '#' 
data = np.loadtxt("fields_2000.dat", comments="#")

# The dump file format is: i j r u v e (six columns)
# Extract the density field "r" which is the third column (index 2)
r_field = data[:, 2]

# Reshape the 1D array into a 2D array using the grid dimensions (Nyh rows, Nxh columns)
r_field_reshaped = r_field.reshape((Nyh, Nxh))

# Create the plot
plt.figure(figsize=(8, 8))
# Display the density field image; using 'origin' set to 'lower' to match grid coordinates.
im = plt.imshow(r_field_reshaped, origin='lower', cmap='viridis')
plt.title("Density Field (r)")
plt.xlabel("i index")
plt.ylabel("j index")
plt.colorbar(im, label="Density (r)")

# Save the plot as an image file. This will create a PNG file in the working directory.
plt.savefig("density_field.png", dpi=300, bbox_inches='tight')

# Optionally display the plot in an interactive window
plt.show()
