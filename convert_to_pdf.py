from fpdf import FPDF

def text_to_pdf(text_file, pdf_file):
    try:
        # Create PDF object
        pdf = FPDF()
        
        # Add a page
        pdf.add_page()
        
        # Set font
        pdf.set_font("Arial", size=12)
        
        # Read text file
        with open(text_file, 'r', encoding='utf-8') as file:
            for line in file:
                # Add text
                pdf.multi_cell(0, 10, txt=line.strip())
        
        # Save the pdf file
        pdf.output(pdf_file)
        print(f"Successfully created {pdf_file}")
        
    except Exception as e:
        print(f"Error creating PDF: {str(e)}")

if __name__ == "__main__":
    text_to_pdf("sample.txt", "sample.pdf")
