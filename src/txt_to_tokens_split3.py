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

temp = str()
for token in doc:
    if token.text != '\n':
        temp += token.text + ' '

n = len(temp)


output = temp[:n//3] + '\n' + temp[n//3:(n//3)*2] + '\n'+ temp[(n//3)*2:]
# Open the output file
with open(output_file_path, "w", encoding="utf-8") as output_file:
    output_file.write(output + "\n")