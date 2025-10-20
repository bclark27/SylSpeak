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


learning_mode = "-l" in sys.argv or "--learn" in sys.argv
en_mode = "-e" in sys.argv or "--english" in sys.argv
space_mode = "-s" in sys.argv or "--space" in sys.argv

chinese_mode = "-c" in sys.argv or "--chinese" in sys.argv
tibet_mode = "-t" in sys.argv or "--tibet" in sys.argv

mode_count = (1 if chinese_mode else 0) + (1 if tibet_mode else 0)
if mode_count > 1:
  print("more then one language mode active")
  exit()

if mode_count == 0:
  chinese_mode = True

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
  alphabet: "abcdefghijklmnopqrstuvwxyz"
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

# Base consonants for main/root letters
base_consonants = {
  'q': 'ྐ',
  'w': 'ྑ',
  'r': 'ྒ',
  't': 'ྔ',
  'y': 'ྕ',
  'p': 'ྖ',
  's': 'ྗ',
  'd': 'ྙ',
  'f': 'ྚ',
  'g': 'ྛ',
  'h': 'ྜ',
  'j': 'ྞ',
  'k': 'ྟ',
  'l': 'ྰ',
  'z': 'ླ',
  'x': 'ྶ',
  'c': 'ྴ',
  'v': 'ྵ',
  'b': 'ྷ',
  'n': 'ྺ',
  'm': 'ྻ',
  '-': 'ྦ',
  '0': 'ི',
  '1': 'ུ',
}

vowels = "aeiou"
# Vowel marks (diacritics)
vowel_marks = {

    '_': 'ྼ',

    'a': 'ཀ',
    'e': 'ཁ',
    'i': 'ག',
    'o': 'གྷ',
    'u': 'ང',

    'aa': 'ཅ',
    'ea': 'ཆ',
    'ia': 'ཇ',
    'oa': 'ཉ',
    'ua': 'ཊ',

    'ae': 'ཋ',
    'ee': 'ཌ',
    'ie': 'ཌྷ',
    'oe': 'ཎ',
    'ue': 'ཏ',
    
    'ai': 'ཐ',
    'ei': 'ད',
    'ii': 'དྷ',
    'oi': 'ན',
    'ui': 'པ',

    'ao': 'ཕ',
    'eo': 'ཬ',
    'io': 'བྷ',
    'oo': 'མ',
    'uo': 'ཙ',

    'au': 'ཚ',
    'eu': 'ཛ',
    'iu': 'ཛྷ',
    'ou': 'ཛྷ',
    'uu': 'ཟ',
}

def split_cv(word):
  
  groups = []
  current = ""
  current_type = None

  for ch in word.lower():
    if ch.isalpha():
      if ch in vowels:
        ch_type = "V"
      else:
        ch_type = "C"

      if current_type == ch_type or current_type is None:
        current += ch
      else:
        groups.append(current)
        current = ch
      current_type = ch_type
  if current:
    groups.append(current)
  return groups

def engToTbt(word):
  word = word.strip()

  groups = split_cv(word)
  
  consts1 = None
  consts2 = None
  vowelGroup = None

  # find the first vowel group, as well as the const group before and after
  for i in range(len(groups)):
    g = groups[i]
    isVowel = g[0] in vowels

    if isVowel and vowelGroup is not None:
      break

    if isVowel:
      vowelGroup = g
      continue
    
    if vowelGroup is None:
      consts1 = g
      continue

    if consts2 is None:
      consts2 = g
      continue
    
    break
  
  endChars = ''
  if consts1 is not None:
    endChars += '0'
  if consts2 is not None:
    endChars += '1'

  midChar = ''
  if consts1 is not None and consts2 is not None:
    midChar = '-'

  if consts1 is None:
    consts1 = ''

  if consts2 is None:
    consts2 = ''

  if vowelGroup is None:
    vowelGroup = '_'

  encoding = ''.join([consts1, midChar, consts2, endChars])
  
  baseChar = vowel_marks[vowelGroup]

  for c in encoding:
    baseChar += base_consonants[c]

  return baseChar

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

      if chinese_mode:
        raw_info[1] = re.sub(r'\[([^\]]+)\]', lambda m: char_dict.get(m.group(1), m.group(1)), raw_info[1])
        char_dict[raw_info[0]] = raw_info[1]
      elif tibet_mode:
        raw_info[1] = "".join([engToTbt(x) for x in raw_info[0].split(' ')]) + "།"

      if en_mode:
        raw_info[1] = f"{raw_info[0]} "
      else:
        if learning_mode:
          raw_info[1] = f"{raw_info[1]}({raw_info[0]})"
        if space_mode:
          raw_info[1] = f"{raw_info[1]} "

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