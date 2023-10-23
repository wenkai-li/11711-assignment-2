import spacy

nlp = spacy.load("en_core_web_sm")

# Input and output file paths
input_file_path = "output_text_file.txt"
output_file_path = "output_tokens.txt"

# Open the input file
with open(input_file_path, "r", encoding="utf-8") as input_file:
    # Read the input file
    text = input_file.read()

doc = nlp(text)

# Open the output file
with open(output_file_path, "w", encoding="utf-8") as output_file:
    for token in doc:
        # Check if the token is not a `\n`
        if token.text != "\n":
            output_file.write(token.text + "\n")