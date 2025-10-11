import os
import shutil
import re
import sys
import subprocess


vocab_path = "vocab.csv"
lang_name = "conlang"

# paths
RIME_ROOT = "/home/ben/.config/ibus/rime"
RIME_BUILD = RIME_ROOT + "/build"


learning_mode = "-h" in sys.argv or "--help" in sys.argv

DEFAULT_CUSTOM_YAML__NAME = "default.custom.yaml"
LANG_SCHEMA_YAML__NAME = f"{lang_name}.schema.yaml"
LANG_DICT_YAML__NAME = f"{lang_name}.dict.yaml"

DEFAULT_CUSTOM_YAML__CONTENT = f"""patch:
  schema_list:
    - schema: {lang_name}
"""

LANG_SCHEMA_YAML__CONTENT = f"""
schema:
  schema_id: {lang_name}
  name: {lang_name}
  version: "1.0"

engine:
  filters:
    - simplifier
    - uniquifier
  processors:
    - ascii_composer
    - recognizer
    - key_binder
    - speller
    - selector
    - navigator
    - express_editor
  segmentors:
    - ascii_segmentor
    - matcher
    - abc_segmentor
    - fallback_segmentor
  translators:
    - "table_translator@custom_phrase"
    - reverse_lookup_translator
    - script_translator

speller:
  alphabet: "abcdefghijklmnopqrstuvwxyz "
  delimiter: " '"

translator:
  dictionary: conlang

menu:
  page_size: 5
"""

LANG_DICT_YAML__CONTENT = f"""
---
name: {lang_name}
version: "1.0"
sort: by_weight
use_preset_vocabulary: false
columns:
  - code
  - text
  - weight
  - stem
...

"""

def create_file(fname, text):
  with open(fname, "w") as file:
    file.write(text)

def clear_directory(path: str):
  """
  Deletes all files and subdirectories in the given directory.

  WARNING: This is irreversible!
  """
  if not os.path.isdir(path):
    raise ValueError(f"{path} is not a valid directory.")

  for entry in os.listdir(path):
    full_path = os.path.join(path, entry)
    if os.path.isfile(full_path) or os.path.islink(full_path):
      os.remove(full_path)  # remove file or symlink
    elif os.path.isdir(full_path):
      shutil.rmtree(full_path)  # remove directory recursively

def clean_rime():
  clear_directory(RIME_BUILD)
  def_file_path = RIME_ROOT + f"/{DEFAULT_CUSTOM_YAML__NAME}"
  if os.path.isfile(def_file_path) or os.path.islink(def_file_path):
    os.remove(def_file_path)

  for name in os.listdir(RIME_ROOT):
    path = os.path.join(RIME_ROOT, name)
    if name.endswith(".userdb") and os.path.isdir(path):
        shutil.rmtree(path)


def parse_vocab(fname):
  
  return vocab

def generate_tab_seperated_vocab():
  vocab_lines = []

  char_dict = {}

  with open(vocab_path, 'r') as file:
    lines = file.readlines()
    for line in lines:
      line = line.split('#', 1)[0].strip(" \n\t")
      
      if line.startswith("#"):
        continue

      if ',' not in line:
        continue

      raw_info = line.split(',')
      raw_info[1] = expanded = re.sub(r'\[([^\]]+)\]', lambda m: char_dict.get(m.group(1), m.group(1)), raw_info[1])
      char_dict[raw_info[0]] = raw_info[1]

      if learning_mode:
        raw_info[1] = f"{raw_info[1]}({raw_info[0]})"

      vocab_lines.append('\t'.join(raw_info))
  return vocab_lines


def main():

  
  vocab_lines = generate_tab_seperated_vocab()
  content = '\n'.join(vocab_lines)
  
  clean_rime()

  create_file(RIME_ROOT + f"/{DEFAULT_CUSTOM_YAML__NAME}", DEFAULT_CUSTOM_YAML__CONTENT)
  create_file(RIME_ROOT + f"/{LANG_SCHEMA_YAML__NAME}", LANG_SCHEMA_YAML__CONTENT)
  create_file(RIME_ROOT + f"/{LANG_DICT_YAML__NAME}", f"{LANG_DICT_YAML__CONTENT}{content}")

  subprocess.run(["ibus", "restart"])

main()