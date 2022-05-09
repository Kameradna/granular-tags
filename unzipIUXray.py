import zipfile

with zipfile.ZipFile("chest-xrays-indiana-university.zip","r") as zip_ref:
    print("Extracting chest-xrays-indiana-university.")
    zip_ref.extractall("IU_xray_data")

with zipfile.ZipFile("reports_and_files.zip","r") as zip_ref:
    print("Extracting reports and files.")
    zip_ref.extractall(".")

print('Done.')