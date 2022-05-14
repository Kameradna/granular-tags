# import os
# from xml_to_dict import XMLtoDict #pip3 install xml_to_dict --user
import pandas as pd

iu_folder_path = 'IU_xray_data'
input_name = 'indiana_reports.csv'
txt_dir = 'txt_reports'
xml_dir = 'xml_reports'

# parser = XMLtoDict()

file_df = pd.read_csv(f'{iu_folder_path}/{input_name}')

for row in file_df.itertuples():
    with open(f'{txt_dir}/{row.uid}.txt','w') as f:
        # f.write(f'uid: {row.uid}\n') #commenting out lines here then rerunning this and the text2bioc allows us to eliminate some of this from the pipeline
        # f.write(f'mesh: {row.MeSH}\n')
        # f.write(f'problems: {row.Problems}\n')
        # f.write(f'image: {row.image}\n')
        # f.write(f'indication: {row.indication}\n')
        # f.write(f'comparison: {row.comparison}\n')
        f.write(f'findings: {row.findings}\n')
        f.write(f'impression: {row.impression}\n')
print('success')




# print(file_df)
# file_df.to_json(f'{iu_folder_path}/{output_name}.json')
# biocjson.dump()
# #convert to json nice with good titles etc then pass to bioc
# #maybe have to format the output into strings or something
# biocxml.dump(,f'{iu_folder_path}/{outputname}.xml')


#use text2bioc from the negbio source
##just split into different text files here
#then use text2bioc

