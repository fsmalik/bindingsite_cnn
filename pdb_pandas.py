import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial.distance import cdist

import os
from pathlib import Path

def process_pdb(pdb_file):
    """
    Converts files from PDB format to a Pandas DataFrame

    Args:
        pdb_file (str): path to .pdb file in string format

    Returns:
        a dataframe named 'pdb': contains all information from pdb file.
    """
    with open(pdb_file, 'r') as initial_read:
        listed_initial_read = list(initial_read)
    
    remark_count = 0

    # Determine the number of remark lines to skip
    for i in np.arange(0, len(listed_initial_read)):
        if (listed_initial_read[i].startswith('REMARK') or 
            listed_initial_read[i].startswith('DBREF') or
            listed_initial_read[i].startswith('SEQADV') or
            listed_initial_read[i].startswith('SEQRES')):
            remark_count = i + 1

    # Define column names for the DataFrame
    column_names = [
        "record_name", "serial_number", "atom_name", "residue", "chain", "res_seq",
        "orth_x", "orth_y", "orth_z", "occupancy", "temp_factor", "element_plus_charge"
    ]

    # Read the PDB file into a DataFrame, skipping remark lines
    pdb = pd.read_csv(pdb_file, sep='\t', skiprows=remark_count, header=None, names=column_names)
    
    # Filter out lines that are not ATOM or HETATM records
    pdbmask = pdb['record_name'].str.startswith('ATOM') | pdb['record_name'].str.startswith('HETATM')
    pdb = pdb[pdbmask]

    # Extract fields from the 'record_name' column
    pdb['serial_number'] = pdb['record_name'].str.slice(start=6, stop=11).str.strip().astype(int)
    pdb['atom_name'] = pdb['record_name'].str.slice(start=11, stop=16).str.strip()
    pdb['residue'] = pdb['record_name'].str.slice(start=16, stop=20).str.strip()
    pdb['chain'] = pdb['record_name'].str.slice(start=20, stop=22).str.strip()
    pdb['res_seq'] = pdb['record_name'].str.slice(start=22, stop=26).str.strip().astype(int)
    pdb['orth_x'] = pdb['record_name'].str.slice(start=30, stop=38).str.strip().astype(float)
    pdb['orth_y'] = pdb['record_name'].str.slice(start=38, stop=46).str.strip().astype(float)
    pdb['orth_z'] = pdb['record_name'].str.slice(start=46, stop=54).str.strip().astype(float)
    pdb['occupancy'] = pdb['record_name'].str.slice(start=54, stop=60).str.strip()
    pdb['temp_factor'] = pdb['record_name'].str.slice(start=60, stop=66).str.strip()
    pdb['element_plus_charge'] = pdb['record_name'].str.slice(start=66, stop=79).str.strip()
    pdb['record_name'] = pdb['record_name'].str.slice(stop=6).str.strip()

    # Remove water and glucose residues
    pdb = pdb[~pdb['residue'].str.startswith('HOH')]  # Removes waters from the PDB
    pdb = pdb[~pdb['residue'].str.startswith('GLC')]  # Removes glucose from the PDB

    return pdb

def bonds_ligand_df(obj, pdb_file=None, sdf_file=None, override_bond_order=False):
    """
    Adds bonds and bond orders to a pdb dataframe of a LIGAND. Requires a SDF file of the ligand with bonding information

    Args:
        obj (object): An object containing:
            - `dataframe` (pandas.DataFrame): Processed PDB data.
            - `filepath` (str): File path to the original PDB file.
        pdb_file (str): path to .pdb file in string format, required only when `override_bond_order=True`.
        sdf_file (str): path to .sdf file in string format, required only when `override_bond_order=True`.
        override_bond_order (boolean): if there is no substructure relations between the .sdf and .pdb this should be set True

    Returns:
        dataframe with bonds and bond orders.
    """
    m = Chem.MolFromPDBFile(pdb_file)
    m2 = Chem.MolFromMolFile(sdf_file)

    filepath = obj.filepath
    df = obj.df
    wd = workingdirectory = Path(filepath).parent
    fn = filename = Path(filepath).stem


    if override_bond_order is True:
        m = Chem.MolFromPDBFile(pdb_file)
        m2 = Chem.MolFromMolFile(sdf_file)
        
        try:
            m_bondorder = AllChem.AssignBondOrdersFromTemplate(m2, m)
        except ValueError as e:
            print(f"Error during bond order assignment: {Path(sdf_file).stem}.sdf. Defaulting to SINGLE bonds.")
            m_bondorder = m

    if override_bond_order is False:
        pdb_file = f"{wd}/{fn}.pdb"
        sdf_file = f"{wd}/{fn}.sdf"
        m = Chem.MolFromPDBFile(pdb_file)
        m2 = Chem.MolFromMolFile(sdf_file)

        try:
            m_bondorder = AllChem.AssignBondOrdersFromTemplate(m2, m)
        except ValueError as e:
            print(f"Error during bond order assignment: {Path(sdf_file).stem}.sdf. Defaulting to SINGLE bonds.")
            m_bondorder = m

        #Chem.Kekulize(m_bondorder, clearAromaticFlags=True)
        #Chem.SanitizeMol(m_bondorder)

        #for atom in m_bondorder.GetAtoms():
        #    total_bond_order = sum(bond.GetBondTypeAsDouble() for bond in atom.GetBonds())
        #    if total_bond_order > 4:
        #        print(fn)
        #        print(f"WARNING: Atom {atom.GetIdx()} ({atom.GetSymbol()}) exceeds valence with bond order {total_bond_order}")

        # Initialize bond columns
        df['bond'] = [[] for _ in range(len(df))]
        df['bond_order'] = [[] for _ in range(len(df))]
    
    rdkit_to_pdb_map = {}
    for atom in m.GetAtoms():
        pdb_info = atom.GetPDBResidueInfo()
        if pdb_info:
            pdb_atom_number = pdb_info.GetSerialNumber()
            rdkit_to_pdb_map[atom.GetIdx()] = pdb_atom_number
    
    
    for bond in m_bondorder.GetBonds():
        bond_type = bond.GetBondType()
        
        # Map bond type to numeric value
        if bond_type == Chem.rdchem.BondType.SINGLE:
            bond_order_value = 1
        elif bond_type == Chem.rdchem.BondType.DOUBLE:
            bond_order_value = 2
        elif bond_type == Chem.rdchem.BondType.AROMATIC:
            bond_order_value = 1.5
        else:
            bond_order_value = 0  # Unknown bond type (not usually the case)
        
        atom_idx1 = bond.GetBeginAtomIdx()
        atom_idx2 = bond.GetEndAtomIdx()

        pdb_atom1 = rdkit_to_pdb_map[atom_idx1]
        pdb_atom2 = rdkit_to_pdb_map[atom_idx2]

        index1 = df.loc[df['serial_number'] == pdb_atom1].index[0]
        index2 = df.loc[df['serial_number'] == pdb_atom2].index[0]

        df.at[index1, 'bond'].append([index1, index2])
        df.at[index1, 'bond_order'].append(bond_order_value)
        df.at[index2, 'bond'].append([index2, index1])
        df.at[index2, 'bond_order'].append(bond_order_value)
    
    hydrogen_atoms = df[df['element_plus_charge'].str.startswith('H')].copy()
    other_atoms = df[~df['element_plus_charge'].str.startswith('H')].copy()

    for _, h_row in hydrogen_atoms.iterrows():
        h_coord = np.array([h_row['orth_x'], h_row['orth_y'], h_row['orth_z']])

        other_atoms_coords = other_atoms[['orth_x', 'orth_y', 'orth_z']].to_numpy()

        distances = cdist([h_coord], other_atoms_coords)
        
        closest_atom_idx = np.argmin(distances)

        df.at[_, 'bond'].append([_, closest_atom_idx])
        df.at[_, 'bond_order'].append(1)
        df.at[closest_atom_idx, 'bond'].append([closest_atom_idx, _])
        df.at[closest_atom_idx, 'bond_order'].append(1)

    return df

def bonds_protein_df(obj):
    """
    Adds bonds and bond orders to a PDB dataframe for a protein structure.

    Args:
        obj (object): An object containing:
            - `dataframe` (pandas.DataFrame): Processed PDB data.
            - `filepath` (str): File path to the original PDB file.

    Returns:
        pandas.DataFrame: Updated dataframe with bond and bond order information.
    """
    filepath = obj.filepath
    df = obj.df
    wd = workingdirectory = Path(filepath).parent
    fn = filename = Path(filepath).stem


    def get_row(atom_name, df=df):
        return df[(df['res_seq'] == res_seq) & (df['atom_name'] == atom_name)]

    def add_bond(idx, target_idx, bond_order):
        df.at[idx, 'bond'].append([idx, target_idx])
        df.at[idx, 'bond_order'].append(bond_order)

    def single_bond(idx, atoms):
        for atom in atoms:
            row = get_row(atom)
            if not row.empty:
                add_bond(idx, row.index[0], 1)

    def aromatic(idx, atoms):
        for atom in atoms:
            row = get_row(atom)
            if not row.empty:
                add_bond(idx, row.index[0], 1.5)

    def double_bond(idx, atoms):
        for atom in atoms:
            row = get_row(atom)
            if not row.empty:
                add_bond(idx, row.index[0], 2)

    def double_local_three(idx, atoms):
        for atom in atoms:
            row = get_row(atom)
            if not row.empty:
                add_bond(idx, row.index[0], 2)

    # Initialize bond columns
    df['bond'] = [[] for _ in range(len(df))]
    df['bond_order'] = [[] for _ in range(len(df))]

    downloaded_residues = set()

    for idx, row in df.iterrows():

        res_seq = row['res_seq']

        if row['record_name'] == 'HETATM': # If your binding site has HETATMs, an SDF file will be downloaded for these bond orders
            
            residue = row['residue']
            
            if len(residue) == 2:
                continue

            if residue in downloaded_residues:
                continue 
            
            # Download the file if not already processed
            url = f"https://files.rcsb.org/ligands/download/{residue}_ideal.sdf"
            newfile = f"{wd}/{fn}_{residue}"
            os.system(f"wget -O {newfile}.sdf {url} > /dev/null 2>&1")
            downloaded_residues.add(residue)
            os.system(f"grep {residue} {filepath} > {newfile}.pdb")
            
            pdb_file1 = f"{newfile}.pdb"
            sdf_file1 = f"{newfile}.sdf"
            
            bonds_ligand_df(obj, pdb_file1, sdf_file1, override_bond_order=True)

        if row['record_name'] == 'ATOM':
            '''
            Backbone atoms connectivity
            '''
            if row['atom_name'] == 'CA': # C-alpha
                row1 = get_row('C')
                row2 = get_row('N')
                row3 = get_row('CB')
                add_bond(idx, row1.index[0], 1)
                add_bond(idx, row2.index[0], 1)
                if not row3.empty:
                    add_bond(idx, row3.index[0], 1)
                if row['residue'] != 'GLY':
                    row4 = get_row('HA') # This is for non-glycines            
                    if not row4.empty:                                                # Non-glycine's hydrogen
                        add_bond(idx, row4.index[0], 1)
                if row['residue'] == 'GLY':
                    row5 = get_row('HA2') # This is for glycines
                    row6 = get_row('HA3') # This is for glycines
                    if not row5.empty:                                                 # Glycine's hydrogens
                        add_bond(idx, row5.index[0], 1)
                        add_bond(row5.index[0], idx, 1)
                        add_bond(idx, row6.index[0], 1)
                        add_bond(row6.index[0], idx, 1)

            if row['atom_name'] == 'C': # Carboxyl Carbon
                row1 = get_row('O')
                add_bond(idx, row1.index[0], 2)
                if (df['res_seq'] == row['res_seq']+1).any():
                    row2 = get_row('N')
                    add_bond(idx, row2.index[0], 1)

            if row['atom_name'] == 'O': # Carboxyl Oxygen
                row1 = get_row('C')
                add_bond(idx, row1.index[0], 2)

            if row['atom_name'] == 'N': # Amino Nitrogen
                row1 = get_row('CA')
                add_bond(idx, row1.index[0], 1)
                row2 = get_row('H')
                if not row2.empty:
                    add_bond(idx, row2.index[0], 1)
                if (df['res_seq'] == row['res_seq']-1).any():
                    row3 = df[(df['res_seq'] == res_seq-1) & (df['atom_name'] == 'C')]
                    if not row2.empty:
                        bond3 = [idx, row2.index[0]]
                        df.at[idx, 'bond'].append(bond3); df.at[idx, 'bond_order'].append(1)
                if row['residue'] == 'PRO':
                    row4 = get_row('CD') # This is only for prolines
                    if not row4.empty:
                        add_bond(idx, row4.index[0], 1)

            if row['atom_name'] == 'H': # Amino Hydrogen
                row1 = get_row('N')
                add_bond(idx, row1.index[0], 1)

            if row['atom_name'] == 'HA': # C-alpha Hydrogen
                row1 = get_row('CA')
                add_bond(idx, row1.index[0], 1)

            if row['atom_name'] in ('HB','HB1','HB2','HB3'): # C-beta hydrogens
                single_bond(idx, ['CB'])
            if row['atom_name'] == 'CB':
                single_bond(idx, ['HB','HB1','HB2','HB3'])

            if row['atom_name'] == 'CB':
                if row['residue'] == 'ALA':
                    single_bond(idx, ['CA'])
                elif row['residue'] == 'VAL':
                    single_bond(idx, ['CG1', 'CG2'])
                elif row['residue'] == 'ILE':
                    single_bond(idx, ['CG1', 'CG2'])
                elif row['residue'] == 'LEU':
                    single_bond(idx, ['CG'])
                elif row['residue'] == 'MET':
                    single_bond(idx, ['CG'])
                elif row['residue'] in ['PHE', 'TYR']:
                    single_bond(idx, ['CG'])
                elif row['residue'] == 'TRP':
                    single_bond(idx, ['CG'])
                elif row['residue'] == 'SER':
                    single_bond(idx, ['OG'])
                elif row['residue'] == 'THR':
                    single_bond(idx, ['OG1', 'CG2'])
                elif row['residue'] == 'ASN':
                    single_bond(idx, ['CG'])
                elif row['residue'] == 'GLN':
                    single_bond(idx, ['CG'])
                elif row['residue'] == 'CYS':
                    single_bond(idx, ['SG'])
                elif row['residue'] == 'PRO':
                    single_bond(idx, ['CG'])
                elif row['residue'] in ['ARG', 'LYS']:
                    single_bond(idx, ['CG'])
                elif row['residue'] == 'ASP':
                    single_bond(idx, ['CG'])
                elif row['residue'] == 'GLU':
                    single_bond(idx, ['CG'])

            '''
            Sidechains
            '''

            if row['residue'] == 'VAL':
                if row['atom_name'] == 'CG1':
                    single_bond(idx, ['CB','HG11','HG12','HG13'])
                if row['atom_name'] == 'CG2':
                    single_bond(idx, ['CB','HG21','HG22','HG23'])
                if row['atom_name'] in ['HG11', 'HG12', 'HG13']:
                    single_bond(idx, ['CG1'])
                if row['atom_name'] in ['HG21', 'HG22', 'HG23']:
                    single_bond(idx, ['CG2'])

            if row['residue'] == 'ILE':
                if row['atom_name'] == 'CG1':
                    single_bond(idx, ['CB','HG12','HG13'])
                if row['atom_name'] == 'CG2':
                    single_bond(idx, ['CB','HG21','HG22','HG23'])
                if row['atom_name'] == 'CD1':
                    single_bond(idx, ['CG1','HD11','HD12','HD13'])
                if row['atom_name'] in ['HG12', 'HG13']:
                    single_bond(idx, ['CG1'])
                if row['atom_name'] in ['HG21', 'HG22', 'HG23']:
                    single_bond(idx, ['CG2'])
                if row['atom_name'] in ['HD11', 'HD12', 'HD13']:
                    single_bond(idx, ['CD1'])

            if row['residue'] == 'LEU':
                if row['atom_name'] == 'CG':
                    single_bond(idx, ['CB','CD1','CD2','HG'])
                if row['atom_name'] == 'CD1':
                    single_bond(idx, ['CG','HD11','HD12','HD13'])
                if row['atom_name'] == 'CD2':
                    single_bond(idx, ['CG','HD21','HD22','HD23'])
                if row['atom_name'] == 'HG':
                    single_bond(idx, ['CG'])
                if row['atom_name'] in ['HD11', 'HD12', 'HD13']:
                    single_bond(idx, ['CD1'])
                if row['atom_name'] in ['HD21', 'HD22', 'HD23']:
                    single_bond(idx, ['CD2'])
                                    
            if row['residue'] == 'MET':
                if row['atom_name'] == 'CG':
                    single_bond(idx, ['CB','SD','HG2','HG3'])
                if row['atom_name'] == 'SD':
                    single_bond(idx, ['CG','CE'])
                if row['atom_name'] == 'CE':
                    single_bond(idx, ['SD','HE1','HE2','HE3'])
                if row['atom_name'] in ['HG2', 'HG3']:
                    single_bond(idx, ['CG'])
                if row['atom_name'] in ['HE1', 'HE2', 'HE3']:
                    single_bond(idx, ['CE'])

            if row['residue'] == 'PHE' or row['residue'] == 'TYR':
                if row['atom_name'] == 'CG':
                    single_bond(idx, ['CB'])
                    aromatic(idx, ['CD1','CD2'])
                if row['atom_name'] == 'CD1':
                    single_bond(idx, ['HD1'])
                    aromatic(idx, ['CG','CE1'])
                if row['atom_name'] == 'CE1':
                    single_bond(idx, ['HE1'])
                    aromatic(idx, ['CD1','CZ'])
                if row['atom_name'] == 'CE2':
                    single_bond(idx, ['HE2'])
                    aromatic(idx, ['CZ','CD2'])
                if row['atom_name'] == 'CD2':
                    single_bond(idx, ['HD2'])
                    aromatic(idx, ['CG','CE2'])
                if row['atom_name'] == 'HD1':
                    single_bond(idx, ['CD1'])
                if row['atom_name'] == 'HE1':
                    single_bond(idx, ['CE1'])
                if row['atom_name'] == 'HE2':
                    single_bond(idx, ['CE2'])
                if row['atom_name'] == 'HD2':
                    single_bond(idx, ['CD2'])

            if row['residue'] == 'PHE':
                if row['atom_name'] == 'CZ':
                    single_bond(idx, ['HZ'])
                    aromatic(idx, ['CE1','CE2'])
                if row['atom_name'] == 'HZ':
                    single_bond(idx, ['CZ'])

            if row['residue'] == 'TYR':
                if row['atom_name'] == 'CZ':
                    single_bond(idx, ['OH'])
                    aromatic(idx, ['CE1','CE2'])
                if row['atom_name'] == 'OH':
                    single_bond(idx, ['HH'])
                if row['atom_name'] == 'HH':
                    single_bond(idx, ['OH'])

            if row['residue'] == 'TRP':
                if row['atom_name'] == 'CG':
                    aromatic(idx, ['CB','CD1','CDD2'])
                if row['atom_name'] == 'CD1':
                    single_bond(idx, ['HD1'])
                    aromatic(idx, ['CG','NE1'])
                if row['atom_name'] == 'NE1':
                    single_bond(idx, ['HE1'])
                    aromatic(idx, ['CD1','CE2'])
                if row['atom_name'] == 'CE2':
                    aromatic(idx, ['NE1','CD2','CZ2'])
                if row['atom_name'] == 'CZ2':
                    single_bond(idx, ['HZ2'])
                    aromatic(idx, ['CE2','CH2'])
                if row['atom_name'] == 'CH2':
                    single_bond(idx, ['HH2'])
                    aromatic(idx, ['CZ2','CZ3'])
                if row['atom_name'] == 'CZ3':
                    single_bond(idx, ['HZ3'])
                    aromatic(idx, ['CH2','CE3'])
                if row['atom_name'] == 'CE3':
                    single_bond(idx, ['HE3'])
                    aromatic(idx, ['CZ3','CD2'])
                if row['atom_name'] == 'CD2':
                    aromatic(idx, ['CE3','CE2','CG'])
                if row['atom_name'] == 'HD1':
                    single_bond(idx, ['CD1'])
                if row['atom_name'] == 'HE1':
                    single_bond(idx, ['NE1'])
                if row['atom_name'] == 'HZ2':
                    single_bond(idx, ['CZ2'])
                if row['atom_name'] == 'HH2':
                    single_bond(idx, ['CH2'])
                if row['atom_name'] == 'HZ3':
                    single_bond(idx, ['CZ3'])
                if row['atom_name'] == 'HE3':
                    single_bond(idx, ['CE3'])

            if row['residue'] == 'SER':
                if row['atom_name'] == 'OG':
                    single_bond(idx, ['CB','HG'])
                if row['atom_name'] == 'HG':
                    single_bond(idx, ['OG'])

            if row['residue'] == 'THR':
                if row['atom_name'] == 'CG2':
                    single_bond(idx, ['CB','HG21','HG22','HG23'])
                if row['atom_name'] == 'OG1':
                    single_bond(idx, ['CB','HG1'])
                if row['atom_name'] == 'HG1':
                    single_bond(idx, ['OG1'])
                if row['atom_name'] in ['HG21', 'HG22', 'HG23']:
                    single_bond(idx, ['CG2'])

            if row['residue'] == 'ASN':
                if row['atom_name'] == 'CG':
                    single_bond(idx, ['CB','ND2'])
                    double_bond(idx, ['OD1'])
                if row['atom_name'] == 'ND2':
                    single_bond(idx, ['CG','HD21','HD22'])
                if row['atom_name'] == 'OD1':
                    double_bond(idx, ['CG'])
                if row['atom_name'] in ['HD21', 'HD22']:
                    single_bond(idx, ['ND2'])

            if row['residue'] == 'GLN':
                if row['atom_name'] == 'CG':
                    single_bond(idx, ['CB','CD','HG2','HG3'])
                if row['atom_name'] == 'CD':
                    single_bond(idx, ['CG','NE2'])
                    double_bond(idx, ['OE1'])
                if row['atom_name'] == 'OE1':
                    double_bond(idx, ['CD'])
                if row['atom_name'] == 'NE2':
                    single_bond(idx, ['CD','HE21','HE22'])
                if row['atom_name'] in ['HG2', 'HG3']:
                    single_bond(idx, ['CG'])
                if row['atom_name'] in ['HE21', 'HE22']:
                    single_bond(idx, ['NE2'])

            if row['residue'] == 'CYS':
                if row['atom_name'] == 'SG':
                    single_bond(idx, ['CB','HG'])
                if row['atom_name'] == 'HG':
                    single_bond(idx, ['SG'])

            if row['residue'] == 'PRO':
                if row['atom_name'] == 'CG':
                    single_bond(idx, ['CB','CD','HG2','HG3'])
                if row['atom_name'] == 'CD':
                    single_bond(idx, ['CG','N','HD2','HD3'])
                if row['atom_name'] in ['HG2', 'HG3']:
                    single_bond(idx, ['CG'])
                if row['atom_name'] in ['HD2', 'HD3']:
                    single_bond(idx, ['CD'])

            if row['residue'] == 'ARG' or row['residue'] == 'LYS':
                if row['atom_name'] == 'CG':
                    single_bond(idx, ['CB','CD','HG2','HG3'])
                if row['atom_name'] == 'CD':
                    single_bond(idx, ['CG','HD2','HD3'])
                if row['atom_name'] in ['HG2', 'HG3']:
                    single_bond(idx, ['CG'])
                if row['atom_name'] in ['HD2', 'HD3']:
                    single_bond(idx, ['CD'])

            if row['residue'] == 'LYS':
                if row['atom_name'] == 'CD':
                    single_bond(idx, ['CE'])
                if row['atom_name'] == 'CE':
                    single_bond(idx, ['CD','NZ','HE2','HE3'])
                if row['atom_name'] == 'NZ':
                    single_bond(idx, ['CE','HZ1','HZ2','HZ3'])
                if row['atom_name'] in ['HE2', 'HE3']:
                    single_bond(idx, ['CE'])
                if row['atom_name'] in ['HZ1', 'HZ2', 'HZ3']:
                    single_bond(idx, ['NZ'])
                    
            if row['residue'] == 'ARG':
                if row['atom_name'] == 'CD':
                    single_bond(idx, ['NE'])
                if row['atom_name'] == 'NE':
                    single_bond(idx, ['CD','HE'])
                    double_local_three(idx, ['CZ'])
                if row['atom_name'] == 'CZ':
                    double_local_three(idx, ['NE','NH1','NH2'])
                if row['atom_name'] == 'NH1':
                    single_bond(idx, ['HH11','HH12'])
                    double_local_three(idx, ['CZ'])
                if row['atom_name'] == 'NH2':
                    single_bond(idx, ['HH21','HH22'])
                    double_local_three(idx, ['CZ'])
                if row['atom_name'] == 'HE':
                    single_bond(idx, ['NE'])
                if row['atom_name'] in ['HH11', 'HH12']:
                    single_bond(idx, ['NH1'])
                if row['atom_name'] in ['HH21', 'HH22']:
                    single_bond(idx, ['NH2'])

            if row['residue'] == 'HIS':
                if row['atom_name'] == 'CG':
                    single_bond(idx, ['CB'])
                    aromatic(idx, ['CD2','ND1'])
                if row['atom_name'] == 'CD2':
                    single_bond(idx, ['HD2'])
                    aromatic(idx, ['CG','NE2'])
                if row['atom_name'] == 'NE2':
                    aromatic(idx, ['CD2','CE1'])
                if row['atom_name'] == 'CE1':
                    single_bond(idx, ['HE1'])
                    aromatic(idx, ['NE2','ND1'])
                if row['atom_name'] == 'ND1':
                    single_bond(idx, ['HD1'])
                    aromatic(idx, ['CE1','CG'])
                if row['atom_name'] == 'HD2':
                    single_bond(idx, ['CD2'])
                if row['atom_name'] == 'HE1':
                    single_bond(idx, ['CE1'])
                if row['atom_name'] == 'HD1':
                    single_bond(idx, ['ND1'])
                    
            if row['residue'] == 'ASP':
                if row['atom_name'] == 'CG':
                    single_bond(idx, ['CB'])
                    aromatic(idx, ['OD1','OD2'])
                if row['atom_name'] == 'OD1':
                    aromatic(idx, ['CG'])
                if row['atom_name'] == 'OD2':
                    aromatic(idx, ['CG'])


            if row['residue'] == 'GLU':
                if row['atom_name'] == 'CG':
                    single_bond(idx, ['CB','CD','HG2','HG3'])
                if row['atom_name'] == 'CD':
                    single_bond(idx, ['CG'])
                    aromatic(idx, ['OE1','OE2'])
                if row['atom_name'] in ['HG2', 'HG3']:
                    single_bond(idx, ['CG'])
                if row['atom_name'] == 'OE1' or row['atom_name'] == 'OE2':
                    aromatic(idx, ['CG'])

    return df