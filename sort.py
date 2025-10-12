vocab_path = "vocab.csv"

def main():

  vocab_lines = []
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
      splitLines.append(raw_info)

    sortedSplitLines = sorted(splitLines, key=lambda x: (len(x[0].split()), x[0].lower()))

    for line in sortedSplitLines:
        vocab_lines.append(','.join(line))

  content = '\n'.join(vocab_lines)
  print(content)

  with open(vocab_path, 'w') as file:
    file.write(content)


main()
