import os
import sys
import json
import copy
import argparse
import nbformat as nbf


CODE_TO_INSERT_1 = """# This cell will require a session restart.
# Accept the restart and continue running notebook cells.
%%capture
import os
!pip install numpy==1.26.4
os.kill(os.getpid(), 9)"""

CODE_TO_INSERT_2 = """%%capture
import os
import sys
from google.colab import drive

# Build Environment.
!pip install pyspharm-syl
!pip install zarr

!apt-get install libproj-dev proj-bin proj-data
!apt-get install libgeos-dev

# shapely must be reinstalled to match geos cartopy
# (https://github.com/SciTools/cartopy/issues/871)
!pip uninstall -y shapely
!pip install --no-binary shapely
!pip install cartopy

# ###############################################################################
# INSTALL MAMBA ON GOOGLE COLAB
# ###############################################################################
! wget -O miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py311_25.11.1-1-Linux-x86_64.sh
! chmod +x miniconda.sh
! bash ./miniconda.sh -b -f -p /usr/local
! rm miniconda.sh
! conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
! conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
! conda config --add channels conda-forge
! conda install -y mamba
! mamba update -qy --all
! mamba clean -qafy
sys.path.append('/usr/local/lib/python3.11/site-packages/')

if os.path.isdir('/content/ufs_model_evaluation_public'):
  !rm -rf /content/ufs_model_evaluation_public

!git clone https://github.com/tariqhamzeygmu/ufs_model_evaluation_public.git

# Install UFS_MODEL_EVALUATION
!mamba env update -n base -f /content/ufs_model_evaluation_public/colab_environment.yml  --yes

basedir = 'ufs_model_evaluation_public'"""

def convert_to_colab(notebook_path: str, target_dir: str) -> None:

    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb_data = nbf.read(f, as_version=4)

    # Every notebook shall start with a "basedir" keyword that indicates where to insert Colab-specific header code.
    insert_new_cells_here = None
    for cell_number in range(len(nb_data['cells'])):

        # Get the cell number of the "basedir" cell.
        this_cell = nb_data['cells'][cell_number]
        this_source = this_cell['source']

        # Here it is:
        if len(this_source) > 0 and this_source.startswith('basedir'):
            if insert_new_cells_here is None:
                insert_new_cells_here = cell_number

        # Clear the output from each cell.  We don't want to make copies of large images.
        nb_data['cells'][cell_number].outputs = []
        nb_data['cells'][cell_number].execution_count = None

    if insert_new_cells_here is None:
        msg = f'Cannot find basedir in source notebook {notebook_path}'
        raise ValueError(msg)

    # Make a copy of the notebook.
    colab_nb = copy.deepcopy(nb_data)

    # Delete the basedir cell
    del colab_nb['cells'][insert_new_cells_here]

    # Update notebook
    cell_1 = nbf.v4.new_code_cell(CODE_TO_INSERT_1)
    cell_2 = nbf.v4.new_code_cell(CODE_TO_INSERT_2)

    colab_nb['cells'].insert(insert_new_cells_here + 0, cell_1)
    colab_nb['cells'].insert(insert_new_cells_here + 1, cell_2)

    # Get this notebook's file name
    file_name = os.path.basename(notebook_path)

    # Construct full target file path
    target_file_path = os.path.join(target_dir, file_name)

    # Write out result to file.
    with open(target_file_path, 'w') as f:
        print(f'Writing: {target_file_path}')
        nbf.write(colab_nb, f)

    return None

if __name__ == '__main__':

    # Parse input arguments
    # iparser = argparse.ArgumentParser()
    # parser.add_argument('--dir')
    # args = parser.parse_args()

    # Current working directory
    cwd = os.path.dirname(os.path.abspath(__file__))

    # Get source notebook directory
    dir_source = os.path.join(cwd, 'enso', 'default')

    # Get target notebook directory
    dir_target = os.path.join(cwd, 'enso', 'colab')

    # Get all files:
    file_list = os.listdir(dir_source) 

    # Precautionary: ensure list contains only .ipynb files.
    file_list = [f for f in file_list if f.split('.')[-1] == 'ipynb']

    # Join the file names with the source directory path.
    file_list = [os.path.join(dir_source, f) for f in file_list]

    # Make each notebook Colab-ready, and write to new files.
    for this_file in file_list:
        convert_to_colab(notebook_path=this_file, target_dir=dir_target)

    sys.exit(0)
