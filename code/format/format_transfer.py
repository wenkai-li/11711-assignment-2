import sys

def process_line(line):
    modified_line = line.replace(" -X- _", "")
    return modified_line

def process_file(input_filename, output_filename):
    i = 0
    with open(input_filename, 'r') as infile, open(output_filename, 'w') as outfile:
        for i, line in enumerate(infile):
            if i == 0:
                continue
            modified_line = process_line(line)
            outfile.write(modified_line)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python format_transfer.py <input_file> <output_file>")
        sys.exit(1)
    input_filename = sys.argv[1]
    output_filename = sys.argv[2]
    process_file(input_filename, output_filename)