vocab_path = "vocab.csv"

def main():

  vocab_lines = []
  with open(vocab_path, 'r') as file:
    lines = file.readlines()
    splitLines = []
    for line in lines:
      line = line.strip()
      splitLines.append(line.split(','))

    sortedSplitLines = sorted(splitLines, key=lambda x: (len(x[0].split()), x[0].lower(), int(x[2])))

    for line in sortedSplitLines:
        vocab_lines.append(','.join(line))

  content = '\n'.join(vocab_lines)
  
  with open(vocab_path, 'w') as file:
    file.write(content)


main()