import numpy as np
import pandas as pd

def process_pdb(pdbfile):
    # Read the initial PDB file
    with open(pdbfile, 'r') as initial_read:
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
    pdb = pd.read_csv(pdbfile, sep='\t', skiprows=remark_count, header=None, names=column_names)
    
    # Filter out lines that are not ATOM or HETATM records
    pdbmask = pdb['record_name'].str.startswith('ATOM') | pdb['record_name'].str.startswith('HETATM')
    pdb = pdb[pdbmask]

    # Extract fields from the 'record_name' column
    pdb['serial_number'] = pdb['record_name'].str.slice(start=6, stop=11).str.strip()
    pdb['atom_name'] = pdb['record_name'].str.slice(start=11, stop=16).str.strip()
    pdb['residue'] = pdb['record_name'].str.slice(start=16, stop=20).str.strip()
    pdb['chain'] = pdb['record_name'].str.slice(start=20, stop=22).str.strip()
    pdb['res_seq'] = pdb['record_name'].str.slice(start=22, stop=26).str.strip()
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
