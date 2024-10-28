# Binding Site Tensor
Contains scripts for a working project that converts protein binding sites to tensors for use in Machine Learning and Deep Learning, particularly designed for 3D CNN.

The Keras code is written in CPU form, but was meant to be converted to GPU since the calculations are intensive. I currently only have MacOS, and this is the reason the project is currently on hold.

However, certain scripts in the program can still be used for other things.

__Requirements:__

python:
  * ```numpy```
  * ```pandas```

shell:
  * ```curl```
    
## Finding Binding Site of PDB(s)
### 1. Isolating a ligand and its corresponding binding site of a single protein structure file
  
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
  
* After running the program you will find a file for the binding site and a file for the ligand, both in ```.pdb``` format. The binding site contains the full residues that are within the given distance of any atom of the ligand. 3.5 Å is the default distance used. 

![8e4l_binding_site_img](https://github.com/user-attachments/assets/de743252-4907-4842-81b6-0153a22d0cb1)
![8e4l_ligand_img](https://github.com/user-attachments/assets/35a88f4a-b73c-4f52-b2a4-debdebd9420a)
![8e4l_combo_img](https://github.com/user-attachments/assets/3936622a-ef3e-47d1-a1a0-97d89b39b4f0)

<div style="display: flex; justify-content: space-around;">
    <img src="(https://github.com/user-attachments/assets/de743252-4907-4842-81b6-0153a22d0cb1)" alt="Image 1" width="300" />
    <img src="(https://github.com/user-attachments/assets/35a88f4a-b73c-4f52-b2a4-debdebd9420a)" alt="Image 2" width="300" />
    <img src="(https://github.com/user-attachments/assets/3936622a-ef3e-47d1-a1a0-97d89b39b4f0)" alt="Image 3" width="300" />
</div>


### 2. Isolating a ligand and its corresponding binding site of a more than one protein structure file

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

* Useful output information that is generated can be written to a ```.log``` file by adding ```>``` at the end of the command.

## Additional Scripts

1. ```process_pdb.py``` a function that imports a ```.pdb``` format file into a dataframe using Pandas.
   * In python: ```df = process_pdb('filename')``` - ```df``` would be the DataFrame's name
   * Removes all waters and glucose. Retains only the ```ATOM``` and ```HETATM``` records from the file.

2. If you used the ```hetatm_batch_script_2.0.sh``` script to generate a number of binding site and ligand files you will have multiple files with extension ```_binding_site.pdb``` and ```_ligand.pdb```. These files can be combined again, correspondingly using the ```combine_ligand+bs.sh``` script.
   * This essentially looks at all the ```.pdb``` files in the current directory and combines ligand and binding site based on the original PDB. For instance, it will take ```7yxr_binding_site.pdb``` and ```7yxr_ligand.pdb``` and combine the two into a single file ```7yxr_combo.pdb```.
   * If the combined file already exists, it will be deleted before making a new version. So if ```7yxr_combo.pdb``` already exisits in the directory it will be deleted and replaced.
    
4. ```TEMP_voxelizer+keras.py``` __is INCOMPLETE and has not been tested completely__ - Instead I suggest using [PyUUL](https://pyuul.readthedocs.io) for protein voxelization.
   * Contains the python class for voxelization functions and reverse functions.
   * Contains scripts for training a 3D CNN on voxel.
   * Hashing protocol is inspired by the [TorchProteinLibray](https://github.com/lamoureux-lab/TorchProteinLibrary). __Note__: the hashing used here is reversed to align more towards drug development utilities.
   * Ignore the ```list_of_data```, it is a temporary method for importing the data.
   * Needs ```process_pdb.py``` to work
   * __Requirements:__
     * Tensor Flow: ```tensorflow```
     * Scikit-learn: ```sklearn```
