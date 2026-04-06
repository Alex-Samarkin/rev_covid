import csv
from datetime import datetime

missing = []
with open(r'C:\Users\alex_\Documents\rev_covid\covid19_daily.csv', 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['Всего (накоп.)'] == 'MD':
            missing.append(row['Дата'])

print(f'Пропуски COVID-19 ({len(missing)} дней):')
if missing:
    print(f'  Первые: {missing[:10]}')
    print(f'  Последние: {missing[-10:]}')
    dates = [datetime.strptime(d, '%Y-%m-%d') for d in missing]
    ranges = []
    start = dates[0]
    prev = dates[0]
    for d in dates[1:]:
        if (d - prev).days > 1:
            ranges.append((start, prev))
            start = d
        prev = d
    ranges.append((start, prev))
    for r in ranges:
        print(f'  {r[0].strftime("%Y-%m-%d")} — {r[1].strftime("%Y-%m-%d")} ({(r[1]-r[0]).days+1} дней)')
