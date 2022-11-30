import json
import copy


def read_ipynb(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        return json.load(f)


first_notebook = read_ipynb('NimaBrahim.ipynb')
second_notebook = read_ipynb('fab.ipynb')


def write_ipynb(notebook, notebook_path):
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f)


# Saving the resulting notebook
final_notebook = copy.deepcopy(first_notebook)
final_notebook['cells'] = first_notebook['cells'] + second_notebook['cells']

write_ipynb(final_notebook, 'NimaBrahimFabian.ipynb')
