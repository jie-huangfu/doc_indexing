from docling.document_converter import DocumentConverter

source = "./ragas.pdf" 
output_path = "./output.md"

converter = DocumentConverter()
result = converter.convert(source)

markdown_text = result.document.export_to_markdown()


with open(output_path, "w", encoding="utf-8") as f:
    f.write(markdown_text)

print(f"Markdown file saved to: {output_path}")

