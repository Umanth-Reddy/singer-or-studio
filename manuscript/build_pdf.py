"""Export the editable Markdown to a line-numbered review PDF."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
from xml.etree import ElementTree

import markdown
from PIL import Image as PillowImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

HERE = Path(__file__).resolve().parent
TEXT = colors.HexColor("#202020")
AUX = colors.HexColor("#4c4c4c")
RULE = colors.HexColor("#c3c3c3")
MARGIN = 47
WIDTH = A4[0] - 2 * MARGIN - 12
FONT_ROOT = Path("/usr/share/fonts/truetype/dejavu")
TABLE_WIDTHS = (
    (0.20, 0.18, 0.15, 0.47),
    (0.24, 0.28, 0.48),
    (0.15, 0.16, 0.13, 0.09, 0.12, 0.14, 0.21),
    (0.14, 0.18, 0.08, 0.11, 0.10, 0.13, 0.07, 0.19),
    (0.14, 0.17, 0.09, 0.28, 0.32),
)
TABLE_WIDTHS = tuple(tuple(n / sum(row) for n in row) for row in TABLE_WIDTHS)


class NumberedParagraph(Paragraph):
    def drawOn(self, canvas, x, y, _sW=0):
        # Baselines mirror ReportLab's paragraph drawing with a fixed leading.
        _, baseline = canvas.absolutePosition(x, y + self.height - self.style.fontSize)
        canvas.manuscript_lines.extend(baseline - i * self.style.leading for i in range(len(self.blPara.lines)))
        super().drawOn(canvas, x, y, _sW)


def font_setup():
    for name, source in {
        "PaperSerif": "DejaVuSerif.ttf",
        "PaperSerif-Bold": "DejaVuSerif-Bold.ttf",
        "PaperSerif-Italic": "DejaVuSerif-Italic.ttf",
        "PaperSerif-BoldItalic": "DejaVuSerif-BoldItalic.ttf",
        "PaperSans": "DejaVuSans.ttf",
    }.items():
        pdfmetrics.registerFont(TTFont(name, str(FONT_ROOT / source)))
    pdfmetrics.registerFontFamily("PaperSerif", normal="PaperSerif", bold="PaperSerif-Bold",
                                 italic="PaperSerif-Italic", boldItalic="PaperSerif-BoldItalic")


def styles():
    base = ParagraphStyle("body", fontName="PaperSerif", fontSize=10, leading=12.5,
                          spaceAfter=5.5, alignment=TA_LEFT, textColor=TEXT)
    return {
        "body": base,
        "title": ParagraphStyle("title", parent=base, fontName="PaperSerif-Bold", fontSize=22,
                                leading=27, spaceAfter=12, keepWithNext=True),
        "h2": ParagraphStyle("h2", parent=base, fontName="PaperSerif-Bold", fontSize=14,
                             leading=18, spaceBefore=10, spaceAfter=5, keepWithNext=True),
        "h3": ParagraphStyle("h3", parent=base, fontName="PaperSerif-Bold", fontSize=10.5,
                             leading=14.1, spaceBefore=7, spaceAfter=4, keepWithNext=True),
        "desc": ParagraphStyle("desc", parent=base, fontName="PaperSans", textColor=AUX,
                               fontSize=8.6, leading=12, spaceAfter=6),
        "caption": ParagraphStyle("caption", parent=base, fontSize=8.9, leading=12.2,
                                  spaceBefore=5, spaceAfter=6, keepWithNext=True),
        "figure_caption": ParagraphStyle("figure_caption", parent=base, fontSize=9.3,
                                         leading=13.3, spaceAfter=12),
        "ref": ParagraphStyle("ref", parent=base, fontSize=8.55, leading=11.15,
                              leftIndent=20, firstLineIndent=-20, spaceAfter=4.6),
        "cell": ParagraphStyle("cell", parent=base, fontName="PaperSans", fontSize=7.55,
                               leading=10.3, spaceAfter=0),
    }


def html_inside(element):
    return (element.text or "") + "".join(
        ElementTree.tostring(child, encoding="unicode") for child in element
    )


def make_table(element, fractions, cell_style, paragraph, vertical_padding=4):
    rows=[]
    for section in element:
        for original in section:
            row=[]
            for cell in original:
                body=html_inside(cell).strip()
                if section.tag=="thead":
                    body=f'<b>{body}</b>'
                row.append(paragraph(body,cell_style))
            rows.append(row)
    if fractions is None:
        fractions = [1 / len(rows[0])] * len(rows[0])
    t=Table(rows,[WIDTH*f for f in fractions],repeatRows=1,hAlign="LEFT", spaceAfter=9)
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#ededed")),
        ("LINEABOVE",(0,0),(-1,0),0.6,AUX),
        ("LINEBELOW",(0,0),(-1,0),0.55,AUX),
        ("LINEBELOW",(0,-1),(-1,-1),0.6,AUX),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f7f7f7")]),
        ("VALIGN",(0,0),(-1,-1),"TOP"),
        ("TOPPADDING",(0,0),(-1,-1),vertical_padding),
        ("BOTTOMPADDING",(0,0),(-1,-1),vertical_padding),
        ("LEFTPADDING",(0,0),(-1,-1),5),
        ("RIGHTPADDING",(0,0),(-1,-1),5),
    ]))
    return t


def decoration(canvas, document, title, thesis=False):
    width,height = A4
    canvas.setTitle(title)
    canvas.setSubject("Production mismatch in singer identity embeddings")
    canvas.setFillColor(AUX)
    canvas.setFont("PaperSans",8)
    canvas.drawString(MARGIN+6,height-31,title.partition("?")[0] + "?")
    canvas.drawRightString(width-MARGIN-6,height-31,
                           "RESEARCH MANUSCRIPT · PROFESSOR REVIEW" if thesis else
                           "RESEARCH ARTICLE")
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.45)
    canvas.line(MARGIN+6,height-39,width-MARGIN-6,height-39)
    canvas.drawRightString(width-MARGIN-6,27,f"{document.page}")
    canvas.manuscript_lines=[]


def add_line_numbers(canvas, _document):
    lines=sorted(canvas.manuscript_lines,reverse=True)
    baselines=[]
    for value in lines:
        if not baselines or baselines[-1]-value > 2.1:
            baselines.append(value)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.setFont("PaperSans",6.9)
    for i, y in enumerate(baselines,1):
        canvas.drawRightString(MARGIN-5,y,str(i))


def create(source, output, thesis=False):
    font_setup()
    st=styles()
    html=markdown.markdown(source.read_text(),extensions=["tables"])
    tree=ElementTree.fromstring(f"<article>{html}</article>")
    title = "".join(tree.find("h1").itertext())
    body=[]
    region=""
    table_number=0
    paragraph = Paragraph if thesis else NumberedParagraph
    thesis_widths = {
        0: TABLE_WIDTHS[0], 1: TABLE_WIDTHS[1], 2: TABLE_WIDTHS[3],
        4: TABLE_WIDTHS[4], 5: (0.15, 0.14, 0.15, *([0.07] * 8)),
        7: TABLE_WIDTHS[2], 13: (0.10, 0.33, 0.12, 0.33, 0.12),
    }
    for element in tree:
        if element.tag=="h1":
            body.append(paragraph(html_inside(element),st["title"]))
        elif element.tag in ("h2","h3"):
            text="".join(element.itertext())
            if element.tag=="h2":
                region=text
                if region in ("Figure captions","Figures"):
                    body.append(PageBreak())
            heading = paragraph(html_inside(element),st[element.tag])
            if thesis and element.tag=="h2":
                heading.outline_title = text
            body.append(heading)
        elif element.tag=="table":
            fractions = thesis_widths.get(table_number) if thesis else TABLE_WIDTHS[table_number]
            column_count = len(element.find("thead/tr"))
            if not thesis and len(fractions) != column_count:
                # The JVS factorial and shared-conditions tables precede the room table.
                fractions = TABLE_WIDTHS[3] if column_count == 8 else None
            padding = 3 if thesis and region.startswith("Appendix") else 4
            body.append(make_table(element,fractions,st["cell"],paragraph,padding))
            table_number+=1
        elif element.tag=="p":
            image=element.find("img")
            if image is not None:
                path=source.parent/image.attrib["src"]
                with PillowImage.open(path) as pic:
                    image_w,image_h=pic.size
                figure = Image(str(path),width=WIDTH,height=WIDTH*image_h/image_w)
                if thesis and body and isinstance(body[-1], Paragraph):
                    body.append(KeepTogether([body.pop(), figure]))
                else:
                    body.append(figure)
                body.append(Spacer(1,10))
                continue
            text=html_inside(element)
            style="body"
            if region=="References":
                style="ref"
                text=re.sub(r'https?://[^\s<]+',lambda m:f'<a href="{m[0]}">{m[0]}</a>',text)
            elif region=="Figure captions":
                style="figure_caption"
            elif text.startswith(("<strong>Table ", "<strong>Figure ")):
                style="caption"
            elif not region:
                style="desc"
            body.append(paragraph(text,st[style]))
        elif thesis and element.tag=="hr":
            body.append(PageBreak())
        else:
            raise ValueError(f"Unsupported source block {element.tag}")
    if not thesis and table_number != len(TABLE_WIDTHS):
        raise ValueError(f"Expected {len(TABLE_WIDTHS)} tables, found {table_number}")
    document=SimpleDocTemplate(str(output),pagesize=A4,leftMargin=MARGIN,rightMargin=MARGIN,
                               topMargin=48,bottomMargin=41,allowSplitting=True)
    # SimpleDocTemplate constructs templates at build time. Use afterPage
    # for the baselines logged by flowables on each page.
    def add_outline(flowable):
        if heading := getattr(flowable,"outline_title",None):
            key = f"heading-{heading}"
            document.canv.bookmarkHorizontalAbsolute(key, document.frame._y)
            document.canv.addOutlineEntry(heading,key)

    if thesis:
        document.afterFlowable = add_outline
    else:
        document.afterPage=lambda: add_line_numbers(document.canv,document)
    def decorate(canvas, doc):
        decoration(canvas, doc, title, thesis=thesis)
    document.build(body,onFirstPage=decorate,onLaterPages=decorate)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=HERE/"manuscript.md")
    parser.add_argument("--output",type=Path,default=HERE/"singer_or_studio_acta_acustica.pdf")
    parser.add_argument("--thesis",action="store_true")
    args=parser.parse_args()
    create(args.source,args.output,thesis=args.thesis)
