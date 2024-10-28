#!/bin/bash

EXT=".pdb"
BS_FILENAME="_binding_site"
L_FILENAME="_ligand"
COMBO_FILENAME="_combo"

for PDB in $(ls -l *.pdb | awk '{print $9}' | cut -c1-4 | awk '!visited[$0]++')
do
	rm ${PDB}${COMBO_FILENAME}${EXT}
	cat ${PDB}${BS_FILENAME}${EXT} > ${PDB}${COMBO_FILENAME}${EXT}
	cat ${PDB}${L_FILENAME}${EXT} >> ${PDB}${COMBO_FILENAME}${EXT} 
done
