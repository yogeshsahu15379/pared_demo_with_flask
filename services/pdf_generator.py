from fpdf import FPDF
import sqlite3
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class PDF(FPDF):
    def __init__(self, title='report'):
        super().__init__(orientation='L', unit='mm', format='A4')
        self.title = title

    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, self.title, ln=True, align='C')
        self.ln(5)

def truncate_text(text, max_chars):
    return (text[:max_chars] + '...') if len(text) > max_chars else text

def generate_pdf_from_table(table_name):
    allowed_tables = ['results','baju_swing_result','hill_march_result','kadamtal_result','tej_march_result']
    if table_name not in allowed_tables:
        raise ValueError("Invalid table name")

    conn = sqlite3.connect('salute_results.db')
    cursor = conn.cursor()
    try:
        cursor.execute(f"SELECT id, timestamp, angle, status, suggestion, screenshot_path FROM {table_name} ORDER BY id DESC LIMIT 10")
        data = cursor.fetchall()
    except Exception as e:
        raise RuntimeError(str(e))
    finally:
        conn.close()
    pdf_title = f"{table_name.replace('_',' ').capitalize()} Report"
    pdf = PDF(title=pdf_title)
    pdf.add_page()
    pdf.set_font("Arial", size=9)

    col_widths = [15, 40, 90, 25, 50, 30]
    row_height = 60
    headers = ["ID", "Timestamp", "Angle", "Status", "Suggestion", "Screenshot"]

    # Header
    pdf.set_fill_color(200, 200, 200)
    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], 10, header, border=1, fill=True)
    pdf.ln()

    for row in data:
        if pdf.get_y() > 180:
            pdf.add_page()
            for i, header in enumerate(headers):
                pdf.cell(col_widths[i], 10, header, border=1, fill=True)
            pdf.ln()

        id, timestamp, angle, status, suggestion, screenshot_path = row
        suggestion = truncate_text(suggestion, 70) if suggestion else ""

        # ✅ Check status for "correct" or "wrong" (case-insensitive)
        status_lower = str(status).strip().lower()
        if "wrong" in status_lower:
            pdf.set_fill_color(255, 200, 200)  # Light red
        elif "correct" in status_lower:
            pdf.set_fill_color(200, 255, 200)  # Light green
        else:
            pdf.set_fill_color(255, 255, 255)  # Default white

        # Row Cells
        pdf.cell(col_widths[0], row_height, str(id), border=1, fill=True)
        pdf.cell(col_widths[1], row_height, str(timestamp), border=1, fill=True)
        pdf.cell(col_widths[2], row_height, str(angle), border=1, fill=True)
        pdf.cell(col_widths[3], row_height, str(status), border=1, fill=True)
        pdf.cell(col_widths[4], row_height, suggestion, border=1, fill=True)

        # Screenshot Cell
        full_path = os.path.join(BASE_DIR, screenshot_path)
        if os.path.exists(full_path):
            x = pdf.get_x()
            y = pdf.get_y()
            pdf.cell(col_widths[5], row_height, '', border=1)
            pdf.image(full_path, x + 2, y + 2, w=col_widths[5] - 4, h=row_height - 4)
        else:
            pdf.cell(col_widths[5], row_height, 'Image Not Found', border=1)

        pdf.ln()

    file_path = os.path.join(BASE_DIR, f"{table_name}_report.pdf")
    pdf.output(file_path)
    return file_path
