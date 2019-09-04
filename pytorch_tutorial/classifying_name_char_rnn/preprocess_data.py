from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os

def findFiles(path): return glob.glob(path)

print(findFiles('data/names/*.txt'))

import unicodedata
import string

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

#Turn a Unicode string to ASCII
def unicode_to_asscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn' and c in all_letters
    )

print(unicode_to_asscii('Ślusàrski'))

# Build the category_lines dictionary, a list of names per language
category_lines = {}
all_categories = []

# Read a file and split into lines
def read_lines(file_name):
    lines = open(file_name, encoding='utf-8').read().strip().split('\n')
    return [unicode_to_asscii(line) for line in lines]


for file_name in findFiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(file_name))[0]
    all_categories.append(category)
    lines = read_lines((file_name))
    category_lines[category] = lines

n_categories = len(all_categories)
