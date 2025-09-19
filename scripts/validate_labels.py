with open("labels_biolip.txt", "r") as infile, open("labels_biolip_cleaned.txt", "w") as outfile:
    for i, line in enumerate(infile):
        line = line.strip()
        if line.startswith(">") or len(line.split()) >= 4:
            outfile.write(line + "\n")
        else:
            print(f"Skipping malformed line {i+1}: {line}")
