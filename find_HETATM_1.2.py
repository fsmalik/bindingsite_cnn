import numpy as np
import pandas as pd
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--inputpdb", required=True, help='input PDB file in .pdb format')
parser.add_argument("-ht", "--hetatm", required=True, help='ligand HET id in PDB')
parser.add_argument("-b", "--bindingsite_output", required=True, help='name for binding site output file')
parser.add_argument("-l", "--ligand_output", required=True, help='name for ligand output file')
parser.add_argument("-d", "--distance", required=False, help='distance from the ligand that will account for the binding site; defualt is 3.5Å')
parser.add_argument("-c", "--center", action='store_true', help='if included, ligand and protein will be centered to (0,0,0)')
args = parser.parse_args()

pdbfile = str(args.inputpdb)
hetatm = str(args.hetatm)
output_pdb_file = str(args.bindingsite_output)
output_ligand_file = str(args.ligand_output)

print('** file name:',pdbfile)
print('** ligand ID:',hetatm)

initial_read  = open(pdbfile, 'r')
listed_initial_read = list(initial_read)

remark_count = 0

for i in np.arange(0, len(listed_initial_read)):
    if listed_initial_read[i].startswith('REMARK'):
        remark_count = i+1
    if listed_initial_read[i].startswith('DBREF'):
        remark_count = i+1
    if listed_initial_read[i].startswith('SEQADV'):
        remark_count = i+1
    if listed_initial_read[i].startswith('SEQRES'):
        remark_count = i+1

column_names = ["record_name", "serial_number", "atom_name", "residue", "chain", "res_seq", "orth_x", "orth_y", "orth_z", "occupancy", "temp_factor", "element_plus_charge"]

pdb = pd.read_csv(pdbfile, sep='\t', skiprows=remark_count, header=None, names = column_names)

pdbmask = pdb['record_name'].str.startswith('ATOM') | pdb['record_name'].str.startswith('HETATM')
pdb = pdb[pdbmask]

pdb['serial_number']=pdb['record_name'].str.slice(start=6,stop=11).str.strip()
pdb['atom_name']=pdb['record_name'].str.slice(start=11,stop=16).str.strip()
pdb['residue']=pdb['record_name'].str.slice(start=16,stop=20).str.strip()
pdb['chain']=pdb['record_name'].str.slice(start=20,stop=22).str.strip()
pdb['res_seq']=pdb['record_name'].str.slice(start=22,stop=26).str.strip()
pdb['orth_x']=pdb['record_name'].str.slice(start=30,stop=38).str.strip().astype(float)
pdb['orth_y']=pdb['record_name'].str.slice(start=38,stop=46).str.strip().astype(float)
pdb['orth_z']=pdb['record_name'].str.slice(start=46,stop=54).str.strip().astype(float)
pdb['occupancy']=pdb['record_name'].str.slice(start=54,stop=60).str.strip()
pdb['temp_factor']=pdb['record_name'].str.slice(start=60,stop=66).str.strip()
pdb['element_plus_charge']=pdb['record_name'].str.slice(start=66,stop=79).str.strip()
pdb['record_name']=pdb['record_name'].str.slice(stop=6).str.strip()

pdb = pdb[~pdb['residue'].str.startswith('HOH')] # Removes waters from the PDB
pdb = pdb[~pdb['residue'].str.startswith('GLC')] # Removes waters from the PDB

if args.center:
    centroid_x = pdb['orth_x'].mean()
    centroid_y = pdb['orth_y'].mean()
    centroid_z = pdb['orth_z'].mean()

    pdb['orth_x'] -= centroid_x
    pdb['orth_y'] -= centroid_y
    pdb['orth_z'] -= centroid_z

# TO DROP
# pdb = pdb[(pdb.residue != 'ACE') | (pdb.atom_name != 'CH3')]

# TO QUERY
# pdb.query("residue=='ACE' and atom_name=='H1'")
# pdb.query("residue=='HIE'")

# here we create separate dataframes for the ligand and protein

ligand_raw = pdb[pdb['residue'].str.startswith(hetatm)]
protein_raw = pdb[~pdb['residue'].str.startswith(hetatm)]

if ligand_raw.empty:
    hetatma = 'A'+hetatm
    hetatmb = 'B'+hetatm
    ligand_raw = pdb[pdb['residue'].str.startswith(hetatma, hetatmb)]
    protein_raw = pdb[~pdb['residue'].str.startswith(hetatma, hetatmb)]

drop_chains = ligand_raw["chain"].unique()[1:]

protein = protein_raw[~protein_raw['chain'].isin(drop_chains)]
ligand = ligand_raw[~ligand_raw['chain'].isin(drop_chains)]

first_unique_res_seq = ligand['res_seq'].unique()[0]
ligand = ligand[ligand['res_seq'] == first_unique_res_seq]

primary_chain = ligand_raw["chain"].unique()[0]

print('** using Chain',primary_chain)

def cartesian_distance(x1, y1, z1, x2, y2, z2):
    """
    Calculate the Cartesian distance between two points in 3D space.
    
    Args:
        x1, y1, z1: Coordinates of the first point.
        x2, y2, z2: Coordinates of the second point.
    
    Returns:
        Distance between the two points.
    """
    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    return distance

bs_res_rows = []
if args.distance:
    distance = float(args.distance)
else:
    distance = 3.5

print('** binding site distance from ligand is:', distance,'Å')

for lrow in ligand.itertuples(index=False):
    pointA = (lrow.orth_x, lrow.orth_y, lrow.orth_z)
    for prow in protein.itertuples(index=False):
        pointB = (prow.orth_x, prow.orth_y, prow.orth_z)
        if cartesian_distance(*pointA, *pointB) <= distance:
            bs_res_rows.append(pd.Series(prow._asdict()))

if not bs_res_rows:
    print("No binding site residues found within the specified distance. Exiting program. Try increasing binding site distance.")
    sys.exit(1) 

bs_atoms = pd.concat(bs_res_rows, axis=1).T

# Need to create a binding site dataframe with the complete residues, not just the atoms
bindingsite_unique = bs_atoms[['residue', 'chain', 'res_seq']].drop_duplicates()
bindingsite = pd.merge(protein, bindingsite_unique, on=['residue', 'chain', 'res_seq'], how='inner')


with open(output_pdb_file, 'w') as f:
    for i, row in bindingsite.iterrows():
        orth_x_formatted = '{:.3f}'.format(row["orth_x"])
        orth_y_formatted = '{:.3f}'.format(row["orth_y"])
        orth_z_formatted = '{:.3f}'.format(row["orth_z"])
        atom_line = f'{row["record_name"]:<6}{row["serial_number"]:>5}{row["atom_name"]:>5}{row["residue"]:>4}{row["chain"]:>2}{row["res_seq"]:>4}{orth_x_formatted:>12}{orth_y_formatted:>8}{orth_z_formatted:>8}{row["occupancy"]:>6}{row["temp_factor"]:>6}{row["element_plus_charge"]:>12}'+'\n'
        f.write(atom_line)

with open(output_ligand_file, 'w') as f:
    for i, row in ligand.iterrows():
        orth_x_formatted = '{:.3f}'.format(row["orth_x"])
        orth_y_formatted = '{:.3f}'.format(row["orth_y"])
        orth_z_formatted = '{:.3f}'.format(row["orth_z"])
        atom_line = f'{row["record_name"]:<6}{row["serial_number"]:>5}{row["atom_name"]:>5}{row["residue"]:>4}{row["chain"]:>2}{row["res_seq"]:>4}{orth_x_formatted:>12}{orth_y_formatted:>8}{orth_z_formatted:>8}{row["occupancy"]:>6}{row["temp_factor"]:>6}{row["element_plus_charge"]:>12}'+'\n'
        f.write(atom_line)

