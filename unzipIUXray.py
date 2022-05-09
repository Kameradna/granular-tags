import zipfile

with zipfile.ZipFile("chest-xrays-indiana-university.zip","r") as zip_ref:
    zip_ref.extractall("IU_xray_data")

