vocab_path = "vocab.csv"
rad_path = "radicals.csv"

def get_radicals():
  with open(rad_path, 'r') as file:
    lines = file.readlines()
    lines = [x.lower().strip() for x in lines]
    return set(lines)

def main():

  all_rads = get_radicals()

  en = {}
  dup_en = {}
  zn = {}
  dup_zn = {}
  rad = {}
  dup_rad = {}
  missing_rads = {}
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
      rank = raw_info[2]

      if e in en:
        dup_en[e] = 0
      if z in zn:
        dup_zn[z] = 0
      en[e] = 0
      zn[z] = 0

      if len(raw_info) >= 4:
        rads = raw_info[3]
        split_rads = rads.split(' ')
        for r in split_rads:
          if r not in all_rads:
            missing_rads[r] = 0
        
        if rads in rad:
          dup_rad[rads] = 0
        rad[rads] = 0


    print("Dup En: ", dup_en)
    print("Dup Zn: ", dup_zn)
    print("Dup Rads: ", dup_rad)
    print("Missing Rads: ", missing_rads)

main()
