import zipfile

with zipfile.ZipFile("chest-xrays-indiana-university.zip","r") as zip_ref:
    print("Extracting chest-xrays-indiana-university.")
    zip_ref.extractall("IU_xray_data")

print('Done.')