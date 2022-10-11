import nbformat
import os 
print(os.getcwd())

source = 'bert_raw_nb'
name = 'bert_inner_work'

nb = nbformat.read(f'notebooks/raw_nb_files/{source}.py',
   nbformat.current_nbformat)

nbformat.write(nb, f'notebooks/{name}.ipynb',
         nbformat.NO_CONVERT)
