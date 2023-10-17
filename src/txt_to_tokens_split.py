import spacy

nlp = spacy.load("en_core_web_sm")

# Input and output file paths
input_file_path = "/Users/jason/Documents/github/11711-assignment-2/data/output.txt"
output_file_path = "/Users/jason/Documents/github/11711-assignment-2/data/output_tokens.txt"

# Open the input file
with open(input_file_path, "r", encoding="utf-8") as input_file:
    # Read the input file
    text = input_file.read()

doc = nlp(text)

temp = []
for token in doc:
    if token.text != '\n':
        temp.append(token.text)

output = ' '.join(temp)
# Open the output file
with open(output_file_path, "w", encoding="utf-8") as output_file:
    output_file.write(output + "\n")