vocab_path = "vocab.csv"

def main():

  en = {}
  dup_en = {}
  zn = {}
  dup_zn = {}
  with open(vocab_path, 'r') as file:
    lines = file.readlines()
    splitLines = []
    for line in lines:
      line = line.split('#', 1)[0].strip(" \n\t")
      
      if line.startswith("#"):
        continue

      if ',' not in line:
        continue

      raw_info = line.split(',')
      
      e = raw_info[0]
      z = raw_info[1]

      if e in en:
        dup_en[e] = 0
      if z in zn:
        dup_zn[z] = 0
      en[e] = 0
      zn[z] = 0

    print(dup_en)
    print(dup_zn)

main()
