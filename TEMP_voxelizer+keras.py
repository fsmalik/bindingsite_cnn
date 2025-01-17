#!/usr/bin/env python
# coding: utf-8



import numpy as np
import tensorflow as tf

elements_hash = {'C': 1, 'O': 2, 'N': 3, 'P': 4, 'S': 5, 'H': 6}
from process_pdb import process_pdb




"""

This is where the most recent code resides.
The data from above is included i.e. alldata variable
Importing of packages is done above too.

"""

list_of_data = [
    "/Users/faisal/tmp/bindingdb_cnn/rcsb_small_testset/combos/8e4l_binding_site.pdb",
    "/Users/faisal/tmp/bindingdb_cnn/rcsb_small_testset/combos/8e4l_ligand.pdb",
    "/Users/faisal/tmp/bindingdb_cnn/rcsb_small_testset/combos/8e4m_binding_site.pdb",
    "/Users/faisal/tmp/bindingdb_cnn/rcsb_small_testset/combos/8e4m_ligand.pdb",
    "/Users/faisal/tmp/bindingdb_cnn/rcsb_small_testset/combos/8gur_binding_site.pdb",
    "/Users/faisal/tmp/bindingdb_cnn/rcsb_small_testset/combos/8gur_ligand.pdb"
]


def process_pdb_file(filepath, hashing):
    """
    Process a PDB file, convert relevant columns to a NumPy array,
    and map element charges using elements_hash.

    Args:
        filepath (str): Path to the PDB file.
        hashing (dict): Dictionary mapping element names to their corresponding values.

    Returns:
        np.ndarray: Processed data array.
    """
    # Process the PDB file
    pdb = process_pdb(filepath)
    
    # Convert relevant columns to NumPy array
    pdb_data = pdb[['orth_x', 'orth_y', 'orth_z', 'element_plus_charge']].to_numpy()
    
    # Map element charges
    # Create a vectorized function for mapping elements
    map_element = np.vectorize(lambda x: hashing.get(x, 7))  # Default to 7 if not found
    pdb_data[:, 3] = map_element(pdb_data[:, 3])
    
    return pdb_data

def process_all_pdb_files(list_of_data, hashing):
    """
    Process a list of PDB files and return a list of processed data arrays.

    Args:
        list_of_data (list of str): List of file paths to the PDB files.
        hash (dict): Dictionary mapping element names to their corresponding values.

    Returns:
        list of np.ndarray: List of processed data arrays.
    """
    alldata = [process_pdb_file(filepath, hashing) for filepath in list_of_data]
    return alldata

#list_of_data = [
#    "/Users/faisal/tmp/bindingdb_cnn/rcsb_small_testset/combos/8e4l_binding_site.pdb",
#    "/Users/faisal/tmp/bindingdb_cnn/rcsb_small_testset/combos/8e4l_ligand.pdb",
#    "/Users/faisal/tmp/bindingdb_cnn/rcsb_small_testset/combos/8e4m_binding_site.pdb",
#    "/Users/faisal/tmp/bindingdb_cnn/rcsb_small_testset/combos/8e4m_ligand.pdb",
#    "/Users/faisal/tmp/bindingdb_cnn/rcsb_small_testset/combos/8gur_binding_site.pdb",
#    "/Users/faisal/tmp/bindingdb_cnn/rcsb_small_testset/combos/8gur_ligand.pdb"
#]

alldata = process_all_pdb_files(list_of_data, elements_hash)




class Voxelizer:
    def __init__(self, datasets, voxel_size=1.0):
        self.datasets = datasets
        self.voxel_size = voxel_size

        # Use the provided global_min and global_max
        self.global_min, self.global_max = self.compute_global_boundaries()

        # Compute voxel grid dimensions
        self.compute_voxel_dimensions()

    def compute_global_boundaries(self):
        # Combine all data points from every dataset to find global min and max
        all_points = np.vstack([data[:, :3] for data in self.datasets])
        global_min = np.min(all_points, axis=0)
        global_max = np.max(all_points, axis=0)
        return global_min, global_max

    def compute_voxel_dimensions(self):
        # Calculate voxel grid dimensions based on global min, max, and voxel size
        self.x_dim = int(np.ceil((self.global_max[0] - self.global_min[0]) / self.voxel_size))
        self.y_dim = int(np.ceil((self.global_max[1] - self.global_min[1]) / self.voxel_size))
        self.z_dim = int(np.ceil((self.global_max[2] - self.global_min[2]) / self.voxel_size))

    def voxelize(self, data):
        data = np.array(data)

        # Shift data so that global_min corresponds to index 0
        shifted_data = data[:, :3] - self.global_min

        # Convert coordinates to voxel indices
        x_indices = np.floor(shifted_data[:, 0] / self.voxel_size).astype(int)
        y_indices = np.floor(shifted_data[:, 1] / self.voxel_size).astype(int)
        z_indices = np.floor(shifted_data[:, 2] / self.voxel_size).astype(int)

        # Ensure that indices are within grid dimensions
        x_indices = np.clip(x_indices, 0, self.x_dim - 1)
        y_indices = np.clip(y_indices, 0, self.y_dim - 1)
        z_indices = np.clip(z_indices, 0, self.z_dim - 1)

        # Create an empty voxel grid
        voxel_grid = np.zeros((self.x_dim, self.y_dim, self.z_dim), dtype=np.float32)
      
        # Assign density values to voxel grid
        for i in range(len(data)):
            voxel_grid[x_indices[i], y_indices[i], z_indices[i]] = 1

        return voxel_grid
    
    def indexer(self, data):
        data = np.array(data)

        # Shift data so that global_min corresponds to index 0
        shifted_data = data[:, :3] - self.global_min

        # Convert coordinates to voxel indices
        x_indices = np.floor(shifted_data[:, 0] / self.voxel_size).astype(int)
        y_indices = np.floor(shifted_data[:, 1] / self.voxel_size).astype(int)
        z_indices = np.floor(shifted_data[:, 2] / self.voxel_size).astype(int)

        # Ensure that indices are within grid dimensions
        x_indices = np.clip(x_indices, 0, self.x_dim - 1)
        y_indices = np.clip(y_indices, 0, self.y_dim - 1)
        z_indices = np.clip(z_indices, 0, self.z_dim - 1)
       
        indicies = []
        
        # Assign density values to voxel grid
        for i in range(len(data)):
            indicies.append([x_indices[i], y_indices[i], z_indices[i]])

        return indicies
    
    def valuer(self, data):
        voxel_values_for_grid = []
        for i in range(len(data)):
            voxel_values_for_grid.append(data[i,3])
        return voxel_values_for_grid
    
    def revert_voxels_to_coordinates(self, voxel_grid):
        # Compute voxel centers
        x_coords = np.linspace(self.global_min[0] + self.voxel_size / 2, 
                               self.global_max[0] - self.voxel_size / 2, self.x_dim)
        y_coords = np.linspace(self.global_min[1] + self.voxel_size / 2, 
                               self.global_max[1] - self.voxel_size / 2, self.y_dim)
        z_coords = np.linspace(self.global_min[2] + self.voxel_size / 2, 
                               self.global_max[2] - self.voxel_size / 2, self.z_dim)
        
        # Get indices where voxel_grid is not zero
        nonzero_indices = np.nonzero(voxel_grid)
        
        # Extract coordinates and corresponding element values of non-zero voxels
        coordinates = []
        for x, y, z in zip(*nonzero_indices):
            coord = [x_coords[x], y_coords[y], z_coords[z]]
            density = voxel_grid[x, y, z]
            coordinates.append(coord + [density])
        
        return np.array(coordinates)


# Example usage:

voxelizer = Voxelizer(alldata, voxel_size=0.5)
input_shape_raw = voxelizer.voxelize(alldata[0]).shape
input_shape_raw

# Voxelize each dataset
# pocket_voxel_grid = voxelizer.voxelize(alldata[0])
# ligand_voxel_grid = voxelizer.voxelize(alldata[1])

# New Jawn
# pocket_value = voxelizer.valuer(alldata[0])
# pocket_index = voxelizer.indexer(alldata[0])




list_of_data[194]




pocket_dense_tensors = []

ligand_dense_tensors = []

for i in range(0,len(alldata[2:])):
    print(i)
    data_value = voxelizer.valuer(alldata[i])
    data_index = voxelizer.indexer(alldata[i])
    data_sparse_tensor = tf.sparse.SparseTensor(indices=data_index, values=data_value, dense_shape=input_shape_raw) # might change this so that it draws the shape from the voxelizer funtion
    data_sparse_tensor = tf.sparse.reorder(data_sparse_tensor)
    dense_tensor = tf.sparse.to_dense(data_sparse_tensor)
    if i % 2 == 0:
        pocket_dense_tensors.append(dense_tensor)
    if i % 2 != 0:
        ligand_dense_tensors.append(dense_tensor)
        
pocket_dense_tensors = tf.stack(pocket_dense_tensors)
ligand_dense_tensors = tf.stack(ligand_dense_tensors)

#temporary configuration of channels
pocket_tensors = tf.expand_dims(pocket_dense_tensors, axis=-1)
ligand_tensors = ligand_dense_tensors




import tensorflow as tf
from tensorflow import keras

input_shape = input_shape_raw +(1,)

model = keras.Sequential([
    keras.Input(shape=input_shape),
    keras.layers.Conv3D(filters=32, kernel_size=(1, 1, 1), activation='relu', data_format="channels_last"),
    keras.layers.Conv3D(filters=8, kernel_size=(1, 1, 1), activation=None, data_format="channels_last")  # No softmax
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(),  # For integer labels
    metrics=[keras.metrics.SparseCategoricalAccuracy()]
)

model.summary()




from sklearn.model_selection import train_test_split
import tensorflow as tf

# Convert TensorFlow tensors to NumPy arrays
pocket_tensors_np = pocket_tensors.numpy()  # Convert to NumPy if it's a Tensor
ligand_tensors_np = ligand_tensors.numpy()  # Convert to NumPy if it's a Tensor

# Split data into training and validation sets (80% training, 20% validation)
pocket_train_np, pocket_val_np, ligand_train_np, ligand_val_np = train_test_split(
    pocket_tensors_np, ligand_tensors_np, test_size=0.2, random_state=42)

# Optionally convert the NumPy arrays back to TensorFlow tensors
pocket_train = tf.convert_to_tensor(pocket_train_np)
pocket_val = tf.convert_to_tensor(pocket_val_np)
ligand_train = tf.convert_to_tensor(ligand_train_np)
ligand_val = tf.convert_to_tensor(ligand_val_np)

# Now you can train the model with validation data
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)




history = model.fit(
    pocket_train, ligand_train,
    epochs=10,
    batch_size=3,
    validation_data=(pocket_val, ligand_val),
    callbacks=[early_stopping]
)




loss, accuracy = model.evaluate(pocket_tensors, ligand_tensors)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")




import matplotlib.pyplot as plt

# Plot training loss
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# If applicable, plot training accuracy
if 'accuracy' in history.history:
    plt.plot(history.history['accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()




# TEST DATA FORMATTING



# TEST POCKET
test_data_value = voxelizer.valuer(alldata[len(alldata)-2])
test_data_index = voxelizer.indexer(alldata[len(alldata)-1])

test_data_sparse_tensor = tf.sparse.SparseTensor(indices=data_index, values=data_value, dense_shape=input_shape_raw) # might change this so that it draws the shape from the voxelizer funtion
test_data_sparse_tensor = tf.sparse.reorder(data_sparse_tensor)

test_dense_tensor = tf.sparse.to_dense(data_sparse_tensor)
        
test_pocket_dense_tensor = test_dense_tensor


#TEST LIGAND
test_data_value = voxelizer.valuer(alldata[5])
test_data_index = voxelizer.indexer(alldata[5])

test_data_sparse_tensor = tf.sparse.SparseTensor(indices=data_index, values=data_value, dense_shape=input_shape_raw) # might change this so that it draws the shape from the voxelizer funtion
test_data_sparse_tensor = tf.sparse.reorder(data_sparse_tensor)

test_dense_tensor = tf.sparse.to_dense(data_sparse_tensor)
        
test_ligand_dense_tensor = test_dense_tensor



test_pocket_tensor = tf.expand_dims(test_pocket_dense_tensor, axis=-1)
test_ligand_tensor = test_ligand_dense_tensor




#prediction = np.argmax(model.predict(test_pocket_tensor), axis=-1)

prediction = model.predict(test_pocket_tensor)





