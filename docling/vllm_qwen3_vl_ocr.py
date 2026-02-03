from pydantic import AnyUrl
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import VlmPipelineOptions
from docling.datamodel.pipeline_options_vlm_model import (
  ApiVlmOptions, ResponseFormat
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline

if __name__ == "__main__":
    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=VlmPipeline,
                pipeline_options=VlmPipelineOptions(
                    enable_remote_services = True,
                    vlm_options = ApiVlmOptions(
                        url=AnyUrl("http://localhost:8000/v1/chat/completions"),
                        params=dict(
                            #model="internvl3_5-4b",
                            model = "Qwen/Qwen3-VL-4B-Instruct",
                            max_tokens=2048,
                            temperature=0.1,
                        ),
                        prompt="OCR the full page to markdown.",
                        timeout=300,
                        scale=0.5,
                        response_format=ResponseFormat.MARKDOWN,
                    )
                )
            )
        }
    )

    result = doc_converter.convert("./pdffiles/label.pdf")
    markdown_content = result.document.export_to_markdown()

    with open("./output/label_content.md", 'w', encoding='utf-8') as f:
        f.write(markdown_content)
