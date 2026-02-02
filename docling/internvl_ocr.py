from pydantic import AnyUrl
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    ApiVlmOptions,
    ResponseFormat,
    VlmPipelineOptions,
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
                        url=AnyUrl("http://localhost:1234/v1/chat/completions"),
                        params=dict(
                            model="internvl3_5-4b",
                            max_tokens=8192,
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

    result = doc_converter.convert("./scanned/label.pdf")
    markdown_content = result.document.export_to_markdown()

    with open("./output/label_content.md", 'w', encoding='utf-8') as f:
        f.write(markdown_content)
