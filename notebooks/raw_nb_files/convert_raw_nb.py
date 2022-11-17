import nbformat
import os 
print(os.getcwd())

source = 'captum_tcav_nb' #'bert_raw_nb'
name = 'captum_tcav' #'bert_inner_work'

nb = nbformat.read(f'notebooks/raw_nb_files/{source}.py',
   nbformat.current_nbformat)

nbformat.write(nb, f'notebooks/{name}.ipynb',
         nbformat.NO_CONVERT)
