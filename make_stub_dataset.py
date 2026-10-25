import numpy as np
import pandas as pd
from datetime import datetime, timedelta

rng = np.random.default_rng(42)
n = 5000

stockcodes = ["POST", "M", "BANK CHARGES", "85123A", "22423", "85099B", "84879"]
countries = ["United Kingdom", "Germany", "France", "EIRE", "Spain", "Netherlands"]
descriptions = ["POSTAGE", "Manual", "Bank Charges", "White Hanging Heart", "Regency Cakestand", "Jumbo Bag Red", "Spotty Bunting"]

start = datetime(2011, 1, 1)
rows = []
for i in range(n):
    code = rng.choice(stockcodes)
    qty = int(rng.integers(1, 30))
    price = float(np.round(rng.uniform(1.0, 60.0), 2))
    rows.append({
        "InvoiceNo": f"{536000 + i // 5}",
        "StockCode": code,
        "Description": descriptions[stockcodes.index(code)],
        "Quantity": qty,
        "InvoiceDate": (start + timedelta(minutes=int(i * 30))).strftime("%m/%d/%Y %H:%M"),
        "UnitPrice": price,
        "CustomerID": int(rng.integers(12346, 18287)) if rng.random() > 0.25 else None,
        "Country": rng.choice(countries, p=[0.85, 0.03, 0.03, 0.03, 0.03, 0.03]),
    })

pd.DataFrame(rows).to_csv("data/OnlineRetail.csv", index=False)
print(f"wrote data/OnlineRetail.csv with {n} rows")
