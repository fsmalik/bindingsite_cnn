#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import math
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--inputpdb", required=True, help='input PDB file in .pdb format')
parser.add_argument("-b", "--bindingsite_output", required=True, help='name for binding site output file in .pdb format')
parser.add_argument("-l", "--ligand_output", required=True, help='name for ligand output file in .pdb format')
parser.add_argument("-d", "--distance", required=False, help='distance from the ligand that will account for the binding site; defualt is 3Å')
parser.add_argument("-m", "--metalions", required=False, action='store_true', help='if you prefer not to include any metal ions in binding site')
parser.add_argument("-c", "--chain", required=False, help='Chain ID; Default is first chain i.e. A')
args = parser.parse_args()


pdbfile = str(args.inputpdb)
output_pdb_file = str(args.bindingsite_output)
output_ligand_file = str(args.ligand_output)

print('** file name:',pdbfile)

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

# TO DROP
# pdb = pdb[(pdb.residue != 'ACE') | (pdb.atom_name != 'CH3')]

# TO QUERY
# pdb.query("residue=='ACE' and atom_name=='H1'")
# pdb.query("residue=='HIE'")


# here we create separate dataframes for the ligand and protein

ligand_raw = pdb[~pdb['record_name'].str.startswith('ATOM')]
protein_raw = pdb[~pdb['record_name'].str.startswith('HETATM')]


# here the metal ion is being removed from the ligand

most_frequent_residue = ligand_raw['residue'].value_counts().idxmax()

metal_ions = ligand_raw[ligand_raw['residue'] != most_frequent_residue]

ligand = ligand_raw[ligand_raw['residue'] == most_frequent_residue]


# here the metal ion is being added to the protein
# this function can be turned off if need be

if args.distance:
    protein = protein_raw
    print('** metal ions will not be included in binding site')
else:
    protein = pd.concat([protein_raw,metal_ions])
    print('** metal ions will be included in binding site')

# this will determine the binding site of the first chain that ligand is in i.e. A
# adding the -c argument will allow to over-ride this and choose your own chain
# if the chain you chose does not have a ligand in it you may not get any results or maybe an error

if args.chain:
    primary_chain = str(args.chain)
else:
    primary_chain = ligand["chain"].iloc[0]

print('** using Chain',primary_chain)

ligand = ligand[ligand['chain']==(primary_chain)]
protein = protein[protein['chain']==(primary_chain)]


def cartesian_distance(x1, y1, z1, x2, y2, z2):
    """
    Calculate the Cartesian distance between two points in 3D space.
    
    Args:
        x1, y1, z1: Coordinates of the first point.
        x2, y2, z2: Coordinates of the second point.
    
    Returns:
        Distance between the two points.
    """
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
    return distance

# to convert to pdb format string -> "{:.3f}".format(hymax)


bs_res_seq = []
if args.distance:
    distance = float(args.distance)
else:
    distance = 3

print('** binding site distance from ligand is:', distance,'Å')

for lrow in ligand.itertuples(index=False):
    pointA = (lrow.orth_x, lrow.orth_y, lrow.orth_z)
    for prow in protein.itertuples(index=False):
        pointB = (prow.orth_x, prow.orth_y, prow.orth_z)
        if cartesian_distance(*pointA, *pointB) <= distance:
            bs_res_seq.append(prow.res_seq)


bindingsite = protein[protein['res_seq'].isin(bs_res_seq)]


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


print("\n")

