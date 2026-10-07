"""Prédictions par classeur. Seules les cellules choisies sont modifiées en OOXML.

Les autres parties du classeur (feuilles, dessins, styles, formules) sont recopiées
à l'identique, plutôt que de réenregistrer tout le document avec un tableur.
"""
import io
import math
import posixpath
import zipfile
import numpy as np
from lxml import etree
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter, column_index_from_string
from threadpoolctl import threadpool_limits

NS = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
REL = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
MAX_PREDICTION_ROWS = 10000


def xml(raw):
    parser = etree.XMLParser(resolve_entities=False, no_network=True)
    root = etree.fromstring(raw, parser)
    if root.getroottree().docinfo.doctype:
        raise ValueError('Le classeur contient une déclaration XML non prise en charge.')
    return root


def check_archive(raw):
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            entries = archive.infolist()
            if (len(entries) > 2000 or len({e.filename for e in entries}) != len(entries)
                    or sum(e.file_size for e in entries) > 100 * 1024 * 1024
                    or any(e.flag_bits & 1 for e in entries)
                    or 'xl/workbook.xml' not in archive.namelist()):
                raise ValueError('Classeur invalide, protégé ou trop volumineux.')
    except zipfile.BadZipFile as exc:
        raise ValueError('Ce fichier n’est pas un classeur .xlsx valide.') from exc


def workbook(raw):
    try:
        return load_workbook(io.BytesIO(raw), read_only=True, data_only=True, keep_links=False)
    except Exception as exc:
        raise ValueError('Impossible de lire le classeur Excel. Choisissez un fichier .xlsx non protégé.') from exc


def describe_workbook(raw, filename, workbook_id):
    check_archive(raw)
    book = workbook(raw)
    try:
        sheets = [dict(name=s.title, rows=s.max_row or 1, columns=s.max_column or 1) for s in book.worksheets]
        if not sheets:
            raise ValueError('Le classeur ne contient aucune feuille.')
        return dict(id=workbook_id, filename=filename, sheets=sheets)
    finally:
        book.close()


def integer(value, lower, upper, label):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f'{label} : choisissez un entier entre {lower} et {upper}.')
    return value


def inspect_sheet(raw, sheet, header_row):
    integer(header_row, 1, 10000, 'Ligne des titres')
    if not isinstance(sheet, str):
        raise ValueError('Choisissez une feuille présente dans le classeur.')
    book = workbook(raw)
    try:
        if sheet not in book.sheetnames:
            raise ValueError('Choisissez une feuille présente dans le classeur.')
        ws = book[sheet]
        if (ws.max_column or 0) > 200:
            raise ValueError('Cette fonction accepte jusqu’à 200 colonnes par feuille.')
        if header_row >= (ws.max_row or 1):
            raise ValueError('Aucune ligne de données après les titres.')
        rows = list(ws.iter_rows(min_row=header_row, max_row=min(ws.max_row, header_row+3), values_only=True))
        columns = [dict(index=i+1, letter=get_column_letter(i+1), name=str(value) if value is not None else '',
                        samples=[str(row[i])[:80] if row[i] is not None else '' for row in rows[1:]])
                   for i, value in enumerate(rows[0])]
        return dict(sheet=sheet, header_row=header_row, columns=columns,
                    first_row=header_row+1, last_row=min(ws.max_row, header_row+MAX_PREDICTION_ROWS),
                    total_rows=ws.max_row)
    finally:
        book.close()


def sheet_path(archive, sheet):
    root = xml(archive.read('xl/workbook.xml'))
    element = next((s for s in root.findall(f'{{{NS}}}sheets/{{{NS}}}sheet') if s.get('name') == sheet), None)
    if element is None:
        raise ValueError('Feuille introuvable.')
    relation_id = element.get(f'{{{REL}}}id')
    rels = xml(archive.read('xl/_rels/workbook.xml.rels'))
    relation = next((r for r in rels if r.get('Id') == relation_id), None)
    if relation is None or relation.get('TargetMode') == 'External':
        raise ValueError('Lien de feuille invalide.')
    target = relation.get('Target', '')
    path = target.lstrip('/') if target.startswith('/') else posixpath.normpath('xl/'+target)
    if path not in archive.namelist() or not path.startswith('xl/'):
        raise ValueError('Contenu de feuille absent.')
    return path, root, rels


def numeric(value):
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return None
    try:
        value = float(value.strip().replace(',', '.')) if isinstance(value, str) else float(value)
        return value if math.isfinite(value) else None
    except (ValueError, OverflowError):
        return None


def predict_workbook(raw, model, features, options):
    if not isinstance(options, dict):
        raise ValueError('Paramètres Excel invalides.')
    sheet, header = options.get('sheet'), options.get('header_row')
    info = inspect_sheet(raw, sheet, header)
    first = integer(options.get('first_row'), header+1, 1048576, 'Première ligne')
    last = integer(options.get('last_row'), first, 1048576, 'Dernière ligne')
    if last-first+1 > MAX_PREDICTION_ROWS:
        raise ValueError('Traitez au maximum 10 000 lignes à la fois.')
    mapping = options.get('mapping')
    if not isinstance(mapping, dict) or set(mapping) != set(features):
        raise ValueError('Associez chaque entrée du modèle à une colonne Excel.')
    columns = [integer(mapping[f], 1, len(info['columns']), 'Colonne d’entrée') for f in features]
    if len(set(columns)) != len(columns):
        raise ValueError('Choisissez une colonne distincte pour chaque entrée.')
    output = integer(options.get('output_column'), 1, len(info['columns'])+1, 'Colonne de sortie')
    if output in columns:
        raise ValueError('La colonne de sortie doit être différente des colonnes d’entrée.')
    name = options.get('output_name', 'Prédiction')
    if not isinstance(name, str) or not 1 <= len(name.strip()) <= 200 or any(ord(c) < 32 for c in name):
        raise ValueError('Donnez un nom à la nouvelle colonne de sortie (200 caractères maximum).')
    overwrite = options.get('overwrite', False)
    if type(overwrite) is not bool:
        raise ValueError('Choisissez si les valeurs existantes doivent être remplacées.')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        path, book_xml, rels = sheet_path(archive, sheet)
        root = xml(archive.read(path))
        for merge in root.findall(f'{{{NS}}}mergeCells/{{{NS}}}mergeCell'):
            from openpyxl.utils.cell import range_boundaries
            left, top, right, bottom = range_boundaries(merge.get('ref'))
            if left <= output <= right and (top <= last and bottom >= first or top <= header <= bottom):
                raise ValueError('La colonne de sortie contient des cellules fusionnées dans cette zone.')
        data = root.find(f'{{{NS}}}sheetData')
        if data is None:
            raise ValueError('La feuille ne contient pas de cellules.')
        rows = {int(row.get('r')):row for row in data}
        cells = {cell.get('r'):cell for row in data for cell in row if etree.QName(cell).localname == 'c'}
        letter = get_column_letter(output)

        def occupied(row):
            cell = cells.get(f'{letter}{row}')
            return cell is not None and (cell.find(f'{{{NS}}}f') is not None
                or any(node.text for node in cell.iter() if etree.QName(node).localname in ('v','t')))

        valid_rows, points, invalid, existing = [], [], [], 0
        book = workbook(raw)
        try:
            for row_number, values in enumerate(book[sheet].iter_rows(min_row=first, max_row=last,
                    max_col=max(columns), values_only=True), first):
                if not overwrite and occupied(row_number):
                    existing += 1
                    continue
                point = [numeric(values[c-1]) for c in columns]
                if any(v is None for v in point):
                    invalid.append(row_number)
                    continue
                valid_rows.append(row_number); points.append(point)
        finally:
            book.close()
        if not points:
            raise ValueError('Aucune ligne à prédire : vérifiez les entrées, les lignes choisies et les cellules de sortie déjà remplies.')
        predictions = []
        with threadpool_limits(limits=1):
            for start in range(0, len(points), 256):
                values = model.predict(np.array(points[start:start+256]))
                if not np.isfinite(values).all():
                    raise ValueError('Le modèle produit une valeur non finie pour ces entrées.')
                predictions.extend(float(v) for v in values)

        def output_cell(row_number):
            ref = f'{letter}{row_number}'
            if ref in cells:
                return cells[ref]
            row = rows.get(row_number)
            if row is None:
                row = etree.Element(f'{{{NS}}}row', r=str(row_number))
                next_row = next((r for r in data if int(r.get('r')) > row_number), None)
                data.insert(data.index(next_row), row) if next_row is not None else data.append(row)
                rows[row_number] = row
            cell = etree.Element(f'{{{NS}}}c', r=ref)
            following = next((c for c in row if etree.QName(c).localname == 'c'
                and column_index_from_string(''.join(ch for ch in c.get('r') if ch.isalpha())) > output), None)
            row.insert(row.index(following), cell) if following is not None else row.append(cell)
            if 'spans' in row.attrib:
                row.set('spans', f'1:{max(output, len(info["columns"]))}')
            cells[ref] = cell
            return cell

        for row_number, value in zip(valid_rows, predictions):
            cell = output_cell(row_number)
            for child in list(cell):
                cell.remove(child)
            cell.attrib.pop('t', None)
            etree.SubElement(cell, f'{{{NS}}}v').text = repr(value)
        if not occupied(header):
            cell = output_cell(header)
            for child in list(cell):
                cell.remove(child)
            cell.set('t','inlineStr')
            etree.SubElement(etree.SubElement(cell, f'{{{NS}}}is'), f'{{{NS}}}t').text = name.strip()
            adjacent = cells.get(f'{get_column_letter(max(1,output-1))}{header}')
            if adjacent is not None and adjacent.get('s') is not None:
                cell.set('s', adjacent.get('s'))
        dimension = root.find(f'{{{NS}}}dimension')
        if dimension is not None:
            dimension.set('ref', f'A1:{get_column_letter(max(output,len(info["columns"])))}{max(info["total_rows"],last)}')
        calc = book_xml.find(f'{{{NS}}}calcPr')
        if calc is None:
            calc = etree.SubElement(book_xml, f'{{{NS}}}calcPr')
        calc.set('fullCalcOnLoad','1'); calc.set('forceFullCalc','1')
        changed = {path:etree.tostring(root), 'xl/workbook.xml':etree.tostring(book_xml)}
        removed = {e.filename for e in archive.infolist() if e.filename == 'xl/calcChain.xml'}
        if removed:
            for relation in list(rels):
                if relation.get('Type','').endswith('/calcChain'):
                    rels.remove(relation)
            types = xml(archive.read('[Content_Types].xml'))
            for item in list(types):
                if item.get('PartName') == '/xl/calcChain.xml':
                    types.remove(item)
            changed.update({'xl/_rels/workbook.xml.rels':etree.tostring(rels), '[Content_Types].xml':etree.tostring(types)})
        result = io.BytesIO()
        with zipfile.ZipFile(result, 'w') as writer:
            for entry in archive.infolist():
                if entry.filename not in removed:
                    writer.writestr(entry, changed.get(entry.filename, archive.read(entry.filename)))
    return result.getvalue(), dict(predicted=len(points), skipped_invalid=len(invalid), skipped_existing=existing,
                                  invalid_rows=invalid[:20], output=letter)
