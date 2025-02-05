from pdb_pandas import *
from pathlib import Path
import itertools
import dgl

elements_hash = {'H': 1, 'C': 2, 'O': 3, 'N': 4, 'P': 5, 'S': 6}

directory = Path("/home/faisal/tmp/bindingdb_cnn/rcsb_small_testset/combos")

# Get all .pdb files in the current directory
files_binding_site = [file for file in directory.glob("*binding_site.pdb") if not file.name.startswith("._")]
files_ligand = [file for file in directory.glob("*ligand.pdb") if not file.name.startswith("._")]

def process_pdb_single(filepath, hashing=elements_hash):
    """
    Process a PDB file, converts it to pdb dataframe,
    and map element charges using elements_hash,
    and adds this column to the dataframe.

    Args:
        filepath (str): Path to the PDB file.
        hashing (dict): Dictionary mapping element names to their corresponding values. Defaults to `elements_hash`.
    Returns:
        np.ndarray: Processed data array.
    """
    pdb = process_pdb(filepath)
    
    # Map element charges
    pdb['hashing'] = pdb['element_plus_charge'].apply(lambda x: hashing.get(x, 7))  # Default to 7 if not found
    
    return pdb

def process_pdb_multi(list_of_data, hashing=elements_hash):
    """
    Process a list of PDB files and return a list of processed data arrays.

    Args:
        list_of_data (list of str): List of file paths to the PDB files.
        hash (dict): Dictionary mapping element names to their corresponding values.

    Returns:
        list of np.ndarray: List of processed data arrays.
    """
    alldata = [ProcessedPDB(str(filepath), process_pdb_single(filepath, hashing)) for filepath in list_of_data]
    return alldata

class ProcessedPDB:
    def __init__(self, filepath, df):
        self.filepath = filepath
        self.df = df


# Add Bonds and Bond Order to dataframes
alldf_binding_site = [
    ProcessedPDB(obj.filepath, bonds_protein_df(obj))
    for obj in process_pdb_multi(files_binding_site)
]

alldf_ligand = [
    ProcessedPDB(obj.filepath, bonds_ligand_df(obj))
    for obj in process_pdb_multi(files_ligand)
]

def dgl_graph(obj_list):
    """
    Converts the data from dataframes to DGL Graphs.

    Args:
        obj_list (list): A list of objects, where each object contains:
            - `df` (pandas.DataFrame): A DataFrame containing PDB data, including columns like 'bond', 'bond_order', 
              'hashing', and 'coords' (orthogonal coordinates).
            - `filepath` (str): The file path to the original PDB file (though not used in graph creation here, included for context).

    Returns:
        graph_list (list): A list of DGL graphs created from the DataFrames, where each graph contains:
            - `g` (dgl.graph): A graph object with nodes and edges, where:
                - Nodes include 'hashing' (node feature) and 'coords' (coordinates of atoms).
                - Edges include 'bond_order' (bond strength/order between atoms).
    """
    graph_list = []
    for obj in obj_list:
        df = obj.df
        all_bonds = list(itertools.chain(*df['bond']))
        all_bonds_transposed = np.transpose(all_bonds)

        g = dgl.graph((all_bonds_transposed[0], all_bonds_transposed[1]))

        hashing = torch.tensor(df['hashing'].to_numpy()).unsqueeze(1).float()
        if g.num_nodes() != len(hashing): # For ions
            g.add_nodes(len(hashing)-g.num_nodes())
        g.ndata['hashing'] = hashing
        g.ndata['coords'] = torch.tensor(df[['orth_x', 'orth_y', 'orth_z']].to_numpy()).float()
        g.edata['bond_order'] = torch.tensor(list(itertools.chain(*df['bond_order']))).float()
        graph_list.append(g)
    return graph_list
    

allg_bs = dgl_graph(alldf_binding_site)
allg_lig = dgl_graph(alldf_ligand)
