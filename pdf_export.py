"""PDF locaux des vues Plotly, capturées en haute résolution par le navigateur."""
import base64
import binascii
import io
from pathlib import Path
from xml.sax.saxutils import escape
import reportlab
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

fonts = Path(reportlab.__file__).parent / 'fonts'
pdfmetrics.registerFont(TTFont('GPVera', str(fonts / 'Vera.ttf')))
pdfmetrics.registerFont(TTFont('GPVeraBold', str(fonts / 'VeraBd.ttf')))


def make_pdf(payload):
    pages = payload.get('pages') if isinstance(payload, dict) else None
    if not isinstance(pages, list) or not 1 <= len(pages) <= 100:
        raise ValueError('Choisissez entre 1 et 100 pages de graphiques à exporter.')
    output = io.BytesIO()
    pdf = canvas.Canvas(output, pagesize=landscape(A4), pageCompression=1)
    pdf.setTitle('Processus-Gaussiens - Graphiques')
    width, height = landscape(A4)
    for index, page in enumerate(pages):
        if not isinstance(page, dict):
            raise ValueError('Page de graphique invalide.')
        title, description, uri = page.get('title'), page.get('description', ''), page.get('image')
        if (not isinstance(title, str) or not 1 <= len(title) <= 500
                or not isinstance(description, str) or len(description) > 2000
                or not isinstance(uri, str) or not uri.startswith('data:image/png;base64,')
                or len(uri) > 12 * 1024 * 1024):
            raise ValueError('Contenu du graphique invalide.')
        try:
            raw = base64.b64decode(uri.split(',', 1)[1], validate=True)
            with Image.open(io.BytesIO(raw)) as image:
                if image.format != 'PNG' or image.width * image.height > 16_000_000:
                    raise ValueError('Image trop volumineuse ou format incorrect.')
                image.verify()
            picture = ImageReader(io.BytesIO(raw))
            iw, ih = picture.getSize()
        except (ValueError, OSError, binascii.Error, Image.DecompressionBombError) as exc:
            raise ValueError('L’image du graphique est invalide.') from exc
        top = height - 32
        for text, font, size, leading, color in (
            (title, 'GPVeraBold', 17, 22, '#172c49'),
            (description, 'GPVera', 9, 13, '#63748c'),
        ):
            paragraph = Paragraph(escape(text), ParagraphStyle('text', fontName=font, fontSize=size,
                                  leading=leading, textColor=colors.HexColor(color)))
            _, ph = paragraph.wrap(width - 64, height)
            top -= ph
            paragraph.drawOn(pdf, 32, top)
            top -= 8
        available_height = top - 42
        if available_height < 160:
            raise ValueError('Le titre ou la description est trop long pour cette page.')
        factor = min((width - 64) / iw, available_height / ih)
        dw, dh = iw * factor, ih * factor
        pdf.drawImage(picture, (width - dw) / 2, 42 + (available_height - dh) / 2,
                      width=dw, height=dh, mask='auto')
        pdf.setFont('GPVera', 8)
        pdf.setFillColor(colors.HexColor('#63748c'))
        pdf.drawString(32, 20, 'Processus-Gaussiens')
        pdf.drawRightString(width - 32, 20, f'{index + 1} / {len(pages)}')
        pdf.showPage()
    pdf.save()
    return output.getvalue()
