import zipfile

with zipfile.ZipFile("Downloads/iu_xray.zip","r") as zip_ref:
    zip_ref.extractall("IU_xray_data_dir")
