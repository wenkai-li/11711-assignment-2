import PyPDF2
import sys

def pdf_to_txt(input_pdf_path, output_txt_path):
    with open(input_pdf_path, 'rb') as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ''

        # Extract text from each page
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()

        # Save the extracted text to a .txt file
        with open(output_txt_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(text)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script_name.py input_pdf_path output_txt_path")
        #python3 src/pdf_to_text_converter.py src/test.pdf src/output_text_file.txt
        sys.exit(1)
    input_pdf_path = sys.argv[1]
    output_txt_path = sys.argv[2]
    pdf_to_txt(input_pdf_path, output_txt_path)