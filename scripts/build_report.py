"""Generate the Assignment 2 HTML report from Markdown."""

from pathlib import Path

import markdown

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_PATH = PROJECT_ROOT / "docs" / "assignment_2_rollout.md"
HTML_PATH = PROJECT_ROOT / "docs" / "assignment_2_rollout.html"


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ADSP 31021 Assignment 2</title>

    <style>
        body {{
            max-width: 1100px;
            margin: 40px auto;
            padding: 0 30px;
            font-family: Arial, Helvetica, sans-serif;
            line-height: 1.6;
            color: #202124;
        }}

        h1, h2, h3 {{
            color: #800000;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}

        th, td {{
            border: 1px solid #cccccc;
            padding: 10px;
            text-align: left;
        }}

        th {{
            background-color: #f2f2f2;
        }}

        code {{
            background-color: #f4f4f4;
            padding: 2px 5px;
            border-radius: 4px;
        }}

        pre {{
            background-color: #f4f4f4;
            padding: 16px;
            overflow-x: auto;
            border-left: 4px solid #800000;
        }}

        img {{
            max-width: 100%;
            height: auto;
        }}
    </style>
</head>

<body>
{content}
</body>
</html>
"""


def main() -> None:
    """Convert the Markdown report into a standalone HTML file."""
    if not MARKDOWN_PATH.exists():
        raise FileNotFoundError(f"Markdown report not found: {MARKDOWN_PATH}")

    markdown_content = MARKDOWN_PATH.read_text(encoding="utf-8")

    html_content = markdown.markdown(
        markdown_content,
        extensions=[
            "fenced_code",
            "tables",
            "toc",
        ],
    )

    HTML_PATH.write_text(
        HTML_TEMPLATE.format(content=html_content),
        encoding="utf-8",
    )

    print("HTML report created successfully.")
    print(f"Source: {MARKDOWN_PATH}")
    print(f"Output: {HTML_PATH}")


if __name__ == "__main__":
    main()
