#!/usr/bin/env python3
"""Render docs/waymo_mask_and_projection_summary.md to PDF (PIL text + images)."""
from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

try:
    from PIL import ImagePdf  # noqa: F401
except Exception:
    pass


def clean_md(s: str) -> str:
    s = re.sub(r'\*\*(.+?)\*\*', r'\1', s)
    s = re.sub(r'`([^`]+)`', r'\1', s)
    s = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', s)
    s = s.replace('∈', ' in ').replace('→', '->').replace('—', '-')
    s = s.replace('–', '-').replace('≈', '~').replace('×', 'x')
    s = s.replace('±', '+/-').replace('²', '2').replace('•', '-')
    s = s.replace('\u202f', ' ')
    return s


def parse_blocks(text: str):
    blocks = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('# '):
            blocks.append(('h1', line[2:].strip())); i += 1; continue
        if line.startswith('## '):
            blocks.append(('h2', line[3:].strip())); i += 1; continue
        if line.startswith('### '):
            blocks.append(('h3', line[4:].strip())); i += 1; continue
        if line.startswith('---'):
            blocks.append(('hr', '')); i += 1; continue
        if line.startswith('!['):
            m = re.match(r'!\[(.*?)\]\((.*?)\)', line)
            if m:
                blocks.append(('img', m.group(1), m.group(2)))
            i += 1; continue
        if line.startswith('```'):
            i += 1
            code = []
            while i < len(lines) and not lines[i].startswith('```'):
                code.append(lines[i]); i += 1
            if i < len(lines):
                i += 1
            blocks.append(('code', '\n'.join(code)))
            continue
        if line.startswith('|') and i + 1 < len(lines) and re.match(
                r'^\|[\s\-:|]+\|$', lines[i + 1]):
            rows = []
            while i < len(lines) and lines[i].startswith('|'):
                row = [c.strip() for c in lines[i].strip('|').split('|')]
                if not re.match(r'^[\s\-:]+$', ''.join(row)):
                    rows.append(row)
                i += 1
            blocks.append(('table', rows))
            continue
        if line.startswith('> '):
            quote = [line[2:]]
            i += 1
            while i < len(lines) and lines[i].startswith('> '):
                quote.append(lines[i][2:]); i += 1
            blocks.append(('quote', ' '.join(quote)))
            continue
        if line.strip() == '':
            i += 1; continue
        if line.startswith('- ') or re.match(r'^\d+\. ', line):
            items = [line]
            i += 1
            while i < len(lines) and (
                    lines[i].startswith('- ')
                    or re.match(r'^\d+\. ', lines[i])
                    or (lines[i].startswith('  ') and lines[i].strip())):
                items.append(lines[i]); i += 1
            blocks.append(('list', items))
            continue
        para = [line]
        i += 1
        while (i < len(lines) and lines[i].strip()
               and not lines[i].startswith(('#', '!', '|', '```', '---', '> '))
               and not lines[i].startswith('- ')
               and not re.match(r'^\d+\. ', lines[i])):
            para.append(lines[i]); i += 1
        blocks.append(('p', ' '.join(para)))
    return blocks


class PdfBuilder:
    def __init__(self, font_path: str):
        # letter size @ 150 dpi
        self.dpi = 150
        self.page_w = int(8.5 * self.dpi)
        self.page_h = int(11 * self.dpi)
        self.margin = int(0.7 * self.dpi)
        self.content_w = self.page_w - 2 * self.margin
        self.font_path = font_path
        self.fonts = {}
        self.pages = []
        self._new_page()

    def font(self, size: int) -> ImageFont.FreeTypeFont:
        if size not in self.fonts:
            self.fonts[size] = ImageFont.truetype(self.font_path, size)
        return self.fonts[size]

    def _new_page(self):
        self.img = Image.new('RGB', (self.page_w, self.page_h), 'white')
        self.draw = ImageDraw.Draw(self.img)
        self.y = self.margin
        self.pages.append(self.img)

    def ensure(self, need: int):
        if self.y + need > self.page_h - self.margin:
            self._new_page()

    def text_width(self, text: str, font: ImageFont.FreeTypeFont) -> int:
        bbox = self.draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0]

    def wrap(self, text: str, font: ImageFont.FreeTypeFont, max_w: int):
        text = clean_md(text)
        lines = []
        for paragraph in text.split('\n'):
            if paragraph == '':
                lines.append('')
                continue
            cur = ''
            for ch in paragraph:
                trial = cur + ch
                if self.text_width(trial, font) <= max_w or not cur:
                    cur = trial
                else:
                    lines.append(cur)
                    cur = ch
            lines.append(cur)
        return lines

    def draw_text_block(self, text, size, color=(17, 17, 17), indent=0, gap_after=8):
        font = self.font(size)
        max_w = self.content_w - indent
        lines = self.wrap(text, font, max_w)
        line_h = size + 6
        for ln in lines:
            self.ensure(line_h)
            self.draw.text((self.margin + indent, self.y), ln, fill=color, font=font)
            self.y += line_h
        self.y += gap_after

    def add_image(self, path: Path, caption: str):
        im = Image.open(path).convert('RGB')
        iw, ih = im.size
        max_w = self.content_w
        max_h = int(3.8 * self.dpi)
        scale = min(max_w / iw, max_h / ih, 1.0)
        # allow upscale a bit for tiny masks
        if max(iw, ih) < 300:
            scale = min(max_w / iw, max_h / ih)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        im = im.resize((nw, nh), Image.NEAREST if max(iw, ih) < 300 else Image.LANCZOS)
        need = nh + (28 if caption else 8)
        self.ensure(need)
        x = self.margin + (self.content_w - nw) // 2
        self.img.paste(im, (x, self.y))
        self.y += nh + 4
        if caption:
            self.draw_text_block(f'图: {caption}', 11, color=(85, 85, 85), gap_after=12)

    def add_code(self, code: str):
        font = self.font(12)
        lines = clean_md(code).split('\n')
        line_h = 18
        pad = 10
        box_h = line_h * max(len(lines), 1) + 2 * pad
        self.ensure(box_h + 10)
        x0, y0 = self.margin, self.y
        self.draw.rectangle(
            [x0, y0, x0 + self.content_w, y0 + box_h],
            fill=(246, 248, 250), outline=(200, 200, 200))
        yy = y0 + pad
        for ln in lines:
            self.draw.text((x0 + 12, yy), ln, fill=(34, 34, 34), font=font)
            yy += line_h
        self.y = y0 + box_h + 12

    def add_table(self, rows):
        if not rows:
            return
        n_cols = max(len(r) for r in rows)
        rows = [r + [''] * (n_cols - len(r)) for r in rows]
        font = self.font(11)
        header_font = self.font(11)
        col_w = self.content_w // n_cols
        row_h = 30
        need = row_h * len(rows) + 8
        self.ensure(need)
        for r_i, row in enumerate(rows):
            for c_i, cell in enumerate(row):
                x0 = self.margin + c_i * col_w
                y0 = self.y + r_i * row_h
                bg = (240, 240, 240) if r_i == 0 else (
                    (250, 250, 250) if r_i % 2 == 0 else (255, 255, 255))
                self.draw.rectangle(
                    [x0, y0, x0 + col_w, y0 + row_h],
                    fill=bg, outline=(200, 200, 200))
                txt = clean_md(cell)
                # truncate if needed
                while self.text_width(txt, font) > col_w - 8 and len(txt) > 3:
                    txt = txt[:-2] + '…'
                tw = self.text_width(txt, header_font if r_i == 0 else font)
                self.draw.text(
                    (x0 + (col_w - tw) // 2, y0 + 8), txt,
                    fill=(20, 20, 20),
                    font=header_font if r_i == 0 else font)
        self.y += need + 8

    def save(self, out_path: Path):
        rgb_pages = [p.convert('RGB') for p in self.pages]
        rgb_pages[0].save(
            out_path, 'PDF', resolution=self.dpi, save_all=True,
            append_images=rgb_pages[1:])


def main():
    root = Path(__file__).resolve().parents[1]
    md_path = root / 'docs' / 'waymo_mask_and_projection_summary.md'
    out_path = root / 'docs' / 'waymo_mask_and_projection_summary.pdf'
    base = md_path.parent
    blocks = parse_blocks(md_path.read_text(encoding='utf-8'))

    font_path = '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'
    pdf = PdfBuilder(font_path)

    for block in blocks:
        kind = block[0]
        if kind == 'h1':
            pdf.y += 8
            pdf.draw_text_block(block[1], 28, gap_after=14)
            pdf.draw.line(
                [(pdf.margin, pdf.y), (pdf.page_w - pdf.margin, pdf.y)],
                fill=(50, 50, 50), width=2)
            pdf.y += 14
        elif kind == 'h2':
            pdf.y += 16
            pdf.draw_text_block(block[1], 20, gap_after=10)
        elif kind == 'h3':
            pdf.y += 10
            pdf.draw_text_block(block[1], 16, gap_after=8)
        elif kind == 'hr':
            pdf.y += 8
            pdf.ensure(20)
            pdf.draw.line(
                [(pdf.margin, pdf.y), (pdf.page_w - pdf.margin, pdf.y)],
                fill=(200, 200, 200), width=1)
            pdf.y += 14
        elif kind == 'p':
            pdf.draw_text_block(block[1], 13, gap_after=10)
        elif kind == 'quote':
            pdf.draw_text_block(block[1], 13, color=(70, 70, 70), indent=20, gap_after=12)
        elif kind == 'list':
            for item in block[1]:
                item = item.strip()
                if item.startswith('- '):
                    item = '- ' + item[2:]
                pdf.draw_text_block(item, 13, indent=12, gap_after=2)
            pdf.y += 6
        elif kind == 'code':
            pdf.add_code(block[1])
        elif kind == 'table':
            pdf.add_table(block[1])
        elif kind == 'img':
            caption, rel = block[1], block[2]
            img_path = (base / rel).resolve()
            if not img_path.exists():
                pdf.draw_text_block(f'[missing image: {rel}]', 12, color=(180, 0, 0))
                continue
            pdf.add_image(img_path, caption)

    pdf.save(out_path)
    print(f'Wrote {out_path} ({out_path.stat().st_size} bytes, {len(pdf.pages)} pages)')


if __name__ == '__main__':
    main()
