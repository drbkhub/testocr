import pypdfium2 as pdfium

def extract_images_from_pdf(pdf_path, output_path, page_index):
    pdf = pdfium.PdfDocument(pdf_path)
    page = pdf.get_page(page_index)
    bitmap = page.render(scale=3)  
    pil_image = bitmap.to_pil()
    pil_image.save(output_path)
    pdf.close()
