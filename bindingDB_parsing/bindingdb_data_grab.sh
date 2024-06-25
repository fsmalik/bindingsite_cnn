#!/bin/bash

# Input file name
input_file=$1

# Extract lines containing "> <Ligand HET ID in PDB>" and "> <PDB ID(s) for Ligand-Target Complex>"
ligand_het_id=$(grep -A1 "> <Ligand HET ID in PDB>" "$input_file" | grep -v ">")
pdb_id=$(grep -A1 "> <PDB ID(s) for Ligand-Target Complex>" "$input_file" | grep -v ">")

# Format the data into columns separated by tabs
paste <(echo "$ligand_het_id") <(echo "$pdb_id")
