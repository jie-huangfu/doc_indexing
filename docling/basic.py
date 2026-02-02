from docling.document_converter import DocumentConverter

converter = DocumentConverter()

result = converter.convert("./ragas.pdf")
markdown_content = result.document.export_to_markdown()


with open("./output.md", "w", encoding="utf-8") as f:
    f.write(markdown_content)



