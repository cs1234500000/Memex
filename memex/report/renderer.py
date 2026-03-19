"""
report/renderer.py
Converts the filled IR (intermediate representation) markdown into:
  - HTML (with inline CSS, self-contained)
  - PDF (via WeasyPrint)
"""

from __future__ import annotations

import pathlib
from datetime import datetime

import markdown


CSS = """
body { font-family: Georgia, serif; max-width: 860px; margin: 40px auto;
       padding: 0 20px; line-height: 1.7; color: #1a1a1a; }
h1 { font-size: 1.8rem; border-bottom: 2px solid #333; padding-bottom: 8px; }
h2 { font-size: 1.3rem; margin-top: 2rem; color: #222; }
h3 { font-size: 1.1rem; color: #444; }
blockquote { border-left: 4px solid #ccc; padding-left: 1rem; color: #555; }
code { background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }
pre  { background: #f4f4f4; padding: 1rem; overflow-x: auto; }
a { color: #0066cc; }
hr { border: none; border-top: 1px solid #ddd; margin: 2rem 0; }
.meta { color: #888; font-size: 0.85rem; }
"""


class ReportRenderer:
    def to_html(self, markdown_text: str, title: str = "Memex Report") -> str:
        """Convert markdown IR to a self-contained HTML string."""
        body = markdown.markdown(
            markdown_text,
            extensions=["extra", "toc", "nl2br"],
        )
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <style>{CSS}</style>
</head>
<body>
{body}
</body>
</html>"""

    def to_pdf(self, html: str, output_path: str | pathlib.Path) -> pathlib.Path:
        """
        Render HTML to PDF using WeasyPrint.
        Returns the path to the written PDF file.
        """
        try:
            from weasyprint import HTML  # optional dependency
        except ImportError as exc:
            raise RuntimeError(
                "weasyprint is required for PDF export. "
                "Install it with: pip install weasyprint"
            ) from exc

        output_path = pathlib.Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        HTML(string=html).write_pdf(str(output_path))
        return output_path

    def save_html(self, html: str, output_path: str | pathlib.Path) -> pathlib.Path:
        output_path = pathlib.Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")
        return output_path
