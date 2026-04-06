import openpyxl
wb1 = openpyxl.load_workbook(r'C:\Users\alex_\Documents\rev_covid\COVID-19 31.03.25.xlsx', data_only=True)
ws = wb1['летальность с 15.12.2020']

# Строка 16 — данные, H=дата, I=всего, J=за сутки
print("Строки 15-20, колонки H-K:")
for i, row in enumerate(ws.iter_rows(min_row=15, max_row=20, min_col=8, max_col=11, values_only=False)):
    vals = [(cell.value, cell.column_letter) for cell in row if cell.value is not None]
    if vals:
        print(f"  Строка {i+15}: {vals}")
