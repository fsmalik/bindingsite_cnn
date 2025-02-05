#!/bin/bash

for FILE in *_ligand.pdb
do
    f=$(basename $FILE .pdb)
    LIGAND=$(head -1 $FILE | awk '{print $4}')
    URL="https://files.rcsb.org/ligands/download/${LIGAND}_ideal.sdf"
    wget -O $f.sdf $URL
done 