# CSV EDA Studio

A Streamlit app that profiles any CSV file and shows interactive charts (Plotly).

## Run it in VS Code

1. Open this folder in VS Code (`File > Open Folder...`).
2. Open a terminal (`Ctrl+` ` ``) and create a virtual environment:

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the app:

   ```bash
   streamlit run app.py
   ```

   It opens at http://localhost:8501. Upload your CSV in the sidebar, or tick
   **Use bundled sample data** to try the included student dataset.

Tip: in VS Code, pick the `.venv` interpreter via `Ctrl+Shift+P > Python: Select Interpreter`.

## What you get

| Tab | Contents |
|---|---|
| Overview | KPI cards, auto-generated insights, summary stats, column types |
| Data Quality | Missing values, duplicate rows, out-of-range percentages |
| Distributions | Histograms with box marginals, violins, categorical bars and donuts |
| Relationships | Correlation heatmap (Pearson/Spearman), scatter explorer with trend line, scatter matrix |
| Target Analysis | Class balance, feature comparison by class, risk curves, threshold precision/recall |
| Outliers | IQR summary and outlier rows per column |
| Data Explorer | Filterable table with CSV download |

Cleaning options (duplicates, invalid percentages, missing values) are in the sidebar and
update every chart immediately.
