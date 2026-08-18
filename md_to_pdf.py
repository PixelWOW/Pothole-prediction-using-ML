"""
md_to_pdf.py
Convert walkthrough.md → walkthrough.pdf using fpdf2 (pure Python, no GTK needed).
"""

import os
import re
from fpdf import FPDF
from fpdf.enums import XPos, YPos

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MD_FILE  = os.path.join(BASE_DIR, "walkthrough.md")
PDF_FILE = os.path.join(BASE_DIR, "walkthrough.pdf")

# ── Colour palette ─────────────────────────────────────────────────────────────
BG        = (13,  17,  23)
ACCENT    = (99,  102, 241)
TEXT      = (201, 209, 217)
SUBTEXT   = (107, 114, 128)
H1_COL    = (99,  102, 241)
H2_COL    = (6,   182, 212)
H3_COL    = (16,  185, 129)
CODE_BG   = (31,  41,  55)
CODE_TEXT = (167, 243, 208)
TABLE_HDR = (31,  41,  55)
TABLE_ALT = (17,  24,  39)
SEP       = (55,  65,  81)


def clean(s: str) -> str:
    """Strip all non-latin-1 characters (emoji, Unicode) so fpdf core fonts work."""
    # Remove markdown image embeds first
    s = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", s)
    # Remove markdown links but keep link text
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)
    # Remove bold/code markers (keep content)
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    s = re.sub(r"`([^`]+)`", r"\1", s)
    # Strip anything outside latin-1 (0x00–0xFF)
    s = s.encode("latin-1", errors="ignore").decode("latin-1")
    return s.strip()


class WalkthroughPDF(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_fill_color(*BG)
        self.rect(0, 0, self.w, 14, style="F")
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(*SUBTEXT)
        self.set_y(4)
        self.cell(0, 6, "PotholeSense — Build Walkthrough", align="C")
        self.ln(6)

    def footer(self):
        self.set_y(-14)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*SUBTEXT)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    # darken the whole page background
    def _add_bg(self):
        self.set_fill_color(*BG)
        self.rect(0, 0, self.w, self.h, style="F")


def render_pdf(md_path: str, pdf_path: str):
    with open(md_path, encoding="utf-8") as f:
        lines = f.readlines()

    pdf = WalkthroughPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=18)

    # Override add_page to always fill background
    original_add_page = pdf.add_page.__func__
    def patched_add_page(self_inner, *a, **kw):
        original_add_page(self_inner, *a, **kw)
        self_inner._add_bg()
    import types
    pdf.add_page = types.MethodType(patched_add_page, pdf)

    pdf.add_page()

    ml = 18
    mr = 18
    cw = pdf.w - ml - mr
    pdf.set_margins(ml, 10, mr)
    pdf.set_x(ml)

    # ── Title block ────────────────────────────────────────────────────────────
    pdf.ln(8)
    pdf.set_font("Helvetica", "B", 26)
    pdf.set_text_color(*ACCENT)
    pdf.cell(cw, 12, "PotholeSense", align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(*TEXT)
    pdf.cell(cw, 7, "Predictive Pothole Formation System  --  Build Walkthrough",
             align="C", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_draw_color(*ACCENT)
    pdf.set_line_width(0.5)
    pdf.line(ml, pdf.get_y() + 3, pdf.w - mr, pdf.get_y() + 3)
    pdf.ln(8)

    in_code  = False
    code_buf = []
    in_table = False
    tbl_buf  = []

    def flush_code():
        nonlocal code_buf
        if not code_buf:
            return
        lh = 4.8
        bh = len(code_buf) * lh + 6
        y0 = pdf.get_y()
        pdf.set_fill_color(*CODE_BG)
        pdf.rect(ml, y0, cw, bh, style="F")
        pdf.set_draw_color(*SEP)
        pdf.set_line_width(0.3)
        pdf.rect(ml, y0, cw, bh)
        pdf.ln(3)
        pdf.set_font("Courier", "", 7.5)
        pdf.set_text_color(*CODE_TEXT)
        for cl in code_buf:
            txt = cl.rstrip().encode("latin-1", errors="ignore").decode("latin-1")
            pdf.set_x(ml + 3)
            pdf.cell(cw - 6, lh, txt, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(3)
        code_buf = []

    def flush_table():
        nonlocal tbl_buf
        if not tbl_buf:
            return
        rows = [r for r in tbl_buf if not re.match(r"^\|[-| :]+\|$", r.strip())]
        if not rows:
            tbl_buf = []
            return
        parsed = []
        for r in rows:
            cells = [c.strip() for c in r.strip().strip("|").split("|")]
            parsed.append(cells)
        n_cols = max(len(r) for r in parsed)
        col_w  = cw / n_cols
        rh = 6.5
        for ri, row in enumerate(parsed):
            is_hdr = ri == 0
            pdf.set_fill_color(*(TABLE_HDR if is_hdr else (TABLE_ALT if ri % 2 == 0 else BG)))
            pdf.set_font("Helvetica", "B" if is_hdr else "", 8.5)
            pdf.set_text_color(*(H2_COL if is_hdr else TEXT))
            for ci in range(n_cols):
                txt = clean(row[ci]) if ci < len(row) else ""
                pdf.set_x(ml + ci * col_w)
                pdf.cell(col_w, rh, txt, border=0, fill=True, new_x=XPos.RIGHT, new_y=YPos.LAST)
            pdf.ln(rh)
            pdf.set_draw_color(*SEP)
            pdf.set_line_width(0.1)
            pdf.line(ml, pdf.get_y(), ml + cw, pdf.get_y())
        pdf.ln(3)
        tbl_buf = []

    def para(text: str):
        """Write a single paragraph of normal text (handles inline marks via clean)."""
        txt = clean(text)
        if not txt:
            return
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(*TEXT)
        pdf.set_x(ml)
        pdf.multi_cell(cw, 5.5, txt, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    for raw in lines:
        line = raw.rstrip("\n")

        # Code fence
        if re.match(r"^````*|^```", line.strip()):
            if not in_code:
                in_code = True
                flush_table()
            else:
                in_code = False
                flush_code()
            continue
        if in_code:
            code_buf.append(line)
            continue

        # Table
        if line.strip().startswith("|"):
            in_table = True
            tbl_buf.append(line)
            continue
        if in_table:
            flush_table()
            in_table = False

        # H1
        if re.match(r"^# ", line):
            flush_code(); flush_table()
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 18)
            pdf.set_text_color(*H1_COL)
            txt = clean(re.sub(r"^# ", "", line))
            pdf.cell(cw, 10, txt, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_draw_color(*ACCENT)
            pdf.set_line_width(0.5)
            pdf.line(ml, pdf.get_y(), pdf.w - mr, pdf.get_y())
            pdf.ln(3)
            continue

        # H2
        if re.match(r"^## ", line):
            pdf.ln(3)
            pdf.set_font("Helvetica", "B", 13)
            pdf.set_text_color(*H2_COL)
            txt = clean(re.sub(r"^## ", "", line))
            pdf.cell(cw, 8, txt, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_draw_color(*H2_COL)
            pdf.set_line_width(0.25)
            pdf.line(ml, pdf.get_y(), ml + 70, pdf.get_y())
            pdf.ln(2)
            continue

        # H3
        if re.match(r"^### ", line):
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(*H3_COL)
            txt = clean(re.sub(r"^### ", "", line))
            pdf.cell(cw, 7, txt, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(1)
            continue

        # H4+
        if re.match(r"^#{4,}", line):
            pdf.ln(1)
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(*TEXT)
            txt = clean(re.sub(r"^#{4,6} ", "", line))
            pdf.set_x(ml)
            pdf.cell(cw, 6, txt, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            continue

        # HR
        if re.match(r"^-{3,}$|^\*{3,}$|^_{3,}$", line.strip()):
            pdf.ln(2)
            pdf.set_draw_color(*SEP)
            pdf.set_line_width(0.3)
            pdf.line(ml, pdf.get_y(), pdf.w - mr, pdf.get_y())
            pdf.ln(4)
            continue

        # Blockquote
        if line.startswith("> "):
            txt = clean(re.sub(r"^> ", "", line))
            pdf.set_draw_color(*ACCENT)
            pdf.set_line_width(1.0)
            pdf.line(ml + 1, pdf.get_y(), ml + 1, pdf.get_y() + 7)
            pdf.set_font("Helvetica", "I", 9.5)
            pdf.set_text_color(*SUBTEXT)
            pdf.set_x(ml + 5)
            pdf.multi_cell(cw - 5, 5.5, txt, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(1)
            continue

        # Bullet
        bm = re.match(r"^(\s*)[*\-+] (.+)$", line)
        if bm:
            indent = len(bm.group(1))
            txt = clean(bm.group(2))
            if not txt:
                continue
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(*TEXT)
            x_off = ml + indent * 2
            pdf.set_x(x_off)
            pdf.cell(5, 5.5, "-")
            pdf.set_x(x_off + 5)
            pdf.multi_cell(cw - indent * 2 - 5, 5.5, txt, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            continue

        # Numbered list
        nm = re.match(r"^\s*(\d+)\. (.+)$", line)
        if nm:
            txt = clean(nm.group(2))
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(*TEXT)
            pdf.set_x(ml)
            pdf.cell(7, 5.5, f"{nm.group(1)}.")
            pdf.set_x(ml + 8)
            pdf.multi_cell(cw - 8, 5.5, txt, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            continue

        # Empty line
        if not line.strip():
            pdf.ln(2)
            continue

        # Normal paragraph (skip pure image lines)
        if re.match(r"^\s*!\[.*?\]\(.*?\)\s*$", line):
            continue

        para(line)

    flush_code()
    flush_table()

    pdf.output(pdf_path)
    print(f"PDF saved  -->  {pdf_path}")
    print(f"Pages: {pdf.page_no()}")


if __name__ == "__main__":
    render_pdf(MD_FILE, PDF_FILE)
