import zipfile

zip_path = r"C:\Users\PG\Downloads\nvda-project.zip"

with zipfile.ZipFile(zip_path, "r") as zip_ref:
    print("FILES INSIDE ZIP:\n")

    for file in zip_ref.namelist():
        print(file)

        import zipfile
import pandas as pd

zip_path = r"C:\Users\PG\Downloads\nvda-project.zip"

with zipfile.ZipFile(zip_path, "r") as zip_ref:

    for file in zip_ref.namelist():

        if file.endswith("/"):
            continue

        print("\n" + "=" * 80)
        print("FILE:", file)
        print("=" * 80)

        try:

            if file.lower().endswith(".csv"):
                df = pd.read_csv(zip_ref.open(file))

            elif file.lower().endswith((".xlsx", ".xls")):
                df = pd.read_excel(zip_ref.open(file))

            else:
                print("Not a CSV/Excel file — skipped.")
                continue

            print("Shape:", df.shape)
            print("\nColumns:")
            print(df.columns.tolist())

            print("\nFirst 5 rows:")
            display(df.head())

            print("\nData types:")
            print(df.dtypes)

            print("\nMissing values:")
            print(df.isnull().sum())

        except Exception as e:
            print("ERROR:", e)