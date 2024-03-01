#!/bin/bash

# Function to display usage instructions
usage() {
    echo "Usage: $(basename "$0") [options]"
    echo "Options:"
    echo "  -h, --help       Show this help message and exit."
    echo "This script will run the find_HETATM.py script on all"
    echo ".pdb files in the directory. Files should not end in"
    echo "*_binding_site.pdb or *_ligand.pdb - these exts are"
    echo "reserved for the output files."
}

if [ "$#" -eq 0 ]; then
	# List of file names with extensions
	all_files=($(ls *.pdb))

	# Will not calculate the binding site calculation on any output files
	files=()
	for file in "${all_files[@]}"; do
	    if [[ $file != *_binding_site.pdb && $file != *_ligand.pdb ]]; then
	        files+=("$file")
	    fi
	done

	# Running python script
	for file in "${files[@]}"; do
	    filename_no_extension=$(basename "$file" .pdb)
	    command="python3 find_HETATM.py -i $file -b ${filename_no_extension}_binding_site.pdb -l ${filename_no_extension}_ligand.pdb"
	    eval "$command"
	done
	exit 0
fi

# Parse command-line options
while getopts ":h" option; do
    case "$option" in
        h | --help)
            usage
            exit 0
            ;;
        :)
            echo "Error: Option -$OPTARG requires an argument."
            usage
            exit 1
            ;;
        \?)
            echo "Error: Invalid option -$OPTARG."
            usage
            exit 1
            ;;
    esac
done

# If invalid options are provided
echo "Error: Invalid options provided."
usage
exit 1

