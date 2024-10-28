# Binding Site Tensor
Contains scripts for an working project that converts protein binding sites to tensors for use in 3D CNN.

The keras code is written in CPU form, but was meant to be converted to GPU since the calculations are intensive. I currently only have MacOS, and this is the reason the project is currently on hold.

Certain scripts in the program can be used for however.

__Requirements:__

python:
  * ```numpy```
  * ```pandas```

shell:
  * ```curl```
    
## Finding Binding Site of PDB(s)
### 1. Isolating a ligand and it's corresponding binding site of a single protein structure file
  
```markdown
python3 find_HETATM_1.2.py -i input.pdb -ht HETATM_ID -b binding_site_output.pdb -l ligand_output.pdb [-d distance] [-c]
```

```python
-i or --inputpdb (required):
Input PDB file in .pdb format.
Example: -i structure.pdb

-ht or --hetatm (required):
Ligand HET ID in the PDB file.
Example: -ht LIG

-b or --bindingsite_output (required):
Name for the binding site output file.
Example: -b bindingsite_output.pdb

-l or --ligand_output (required):
Name for the ligand output file.
Example: -l ligand_output.pdb

-d or --distance (optional):
Distance (in Å) from the ligand that will account for the binding site.
Default value is 3.5 Å.
Example: -d 4.0

-c or --center (optional):
If included, the ligand and protein will be centered to (0, 0, 0).
Example: include -c to center.
```
* ```find_HETATM_1.2.py``` will find the investigational molecule (i.e. drug or exogenous ligand) in a single PDB file and output both the residues of the protein that make up the binding site and the ligand itself.
  
* After running the program you will find a file for the binding site and a file for the ligand, both in ```.pdb``` format. The binding site contains the full residues of any residue that is within the given distance of any atom of the ligand. 3.5 Å is the default distance used. 

### 2. Isolating a ligand and it's corresponding binding site of a more than one protein structure file

```markdown
bash hetatm_batch_script_2.0.sh [options]
```

```bash
Options:

-h, --help:
Show help message and exit.
```

Ensure that you have the required permissions to execute the script. You may need to make it executable with the following command: 
```chmod +x hetatm_batch_script_2.0.sh```

* ```hetatm_batch_script_2.0.sh``` will autonoumsly execute the ```find_HETATM_1.2.py``` program. 

* Using this script requires internet access since it fetches information from [RCSB](https://www.rcsb.org) database. 

* This script will skip any PDB file bound with amino acids (i.e. neurotransmitters) and works only for hetergenous ligands as described above. 

* Useful output information that can potentially be generated in a ```.log``` file by adding ```>``` at the end of the command.

## Additional Scripts

1. ```process_pdb.py``` a function that imports a ```.pdb``` format file into a dataframe using Pandas.
   * In python: ```df = process_pdb('filename')``` - ```df``` would be the DataFrame's name
   * Removes all waters and glucose. Retains only the ```ATOM``` and ```HETATM``` records from the file.
  
    
3. ```TEMP_voxelizer+keras.py``` __is INCOMPLETE and has not been tested completely__ - Instead I suggest using [PyUUL](https://pyuul.readthedocs.io) for protein voxelization.
   * Contains the python class for voxelization functions and reverse functions.
   * Contains scripts for training a 3D CNN on voxel.
   * Hashing protocol is inspired by the [TorchProteinLibray](https://github.com/lamoureux-lab/TorchProteinLibrary). __Note__: the hashing used here is reversed to align more towards drug development utilities.
   * Ignore the ```list_of_data```, it is a temporary method for importing the data.
   * Needs ```process_pdb.py``` to work
   * __Requirements:__
     * Tensor Flow: ```tensorflow```
     * Scikit-learn: ```sklearn```
