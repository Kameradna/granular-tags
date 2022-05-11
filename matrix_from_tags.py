# label_dict = {'img1':[('crazy',1),('wacky',0.5),('interesting',0)],'img2':[('wacky',1),('kooky',1),('cooked',1)],'img3':[]}

import argparse
import random
import os

import bioc
import numpy as np
import json

#adapted radically from the negbio2 label extractor to csv tool
POSITIVE = 1
NEGATIVE = 0
UNCERTAIN = 1

# Misc. constants
UNCERTAINTY = "uncertainty"
NEGATION = "negation"
OBSERVATION = "term"

def category_test(category):#return bool for goodness
    #we shall test the category for goodness for our purposes. At this point (of writing, 3/5/22) we only know how to perform simple string tests.
    #Ideally we would expand this to UMLS system tests, ie whether the category is a UMLS disease or injury etc classified CUI.
    
    #for simplicity we will completely ignore some pretty random categories (not for final submission)
    #when I finetune the UMLS term extraction process this should be fine to delete
    if category == 'Intrauterine growth restriction, metaphyseal dysplasia, adrenal hypoplasia congenita, and genital anomaly syndrome':
        return False
    # for item in category.split():
    #     if any(chr.isdigit() or chr == '^' for chr in item): #if any numeric present
    #         return False
    return True

def load_xml(xml_dir): #generates dictionary of lists with keys being docs, lists being lists of (UMLS terms, negbio finding) tuple pairs
    label_dict = {}
    print("Extracting xml...")
    with open(xml_dir, encoding='utf8') as fp:
        collection = bioc.load(fp)
    print("Parsing xml...")
    for doc in collection.documents:
        doc_label_list = []
        for p in doc.passages:
            for annotation in p.annotations:
                category = annotation.infons[OBSERVATION]
                if category_test(category) == False:
                    # print(f'ignoring {category}')
                    continue #skip categories which are functionally meaningless for us
                # we want only patf, dsyn, neop and anab ['patf', 'dsyn', 'neop', 'anab']
                if annotation.infons['semtype'] not in ['patf', 'dsyn']:#, 'neop', 'anab'
                    print(f"ignoring {annotation.infons[OBSERVATION]}")
                    continue
                print(f'Woah it is {annotation.infons[OBSERVATION]}')
                if NEGATION in annotation.infons:
                    doc_label_list.append((category,NEGATIVE))
                elif UNCERTAINTY in annotation.infons:
                    doc_label_list.append((category,UNCERTAIN))
                else:
                    doc_label_list.append((category,POSITIVE))
        label_dict[doc.id] = doc_label_list
    with open(f'{args.save_dir}/.label_dict_checkpoint.txt', 'w') as f:
        f.write(f'{label_dict}')
    return label_dict

def matrix_from_tags(descriptor_dict):
    unique_tags_list = []
    unique_count = {}
    print("Generating unique UMLS term list...")
    for uid in descriptor_dict: #for docs with findings, add unique UMLS terms to the list of known UMLS terms
        if descriptor_dict[uid] != []:
            for descriptor,classification in descriptor_dict[uid]:
                if descriptor not in unique_tags_list:
                    unique_tags_list.append(descriptor)
                    unique_count[descriptor] = 1
                else:
                    unique_count[descriptor] += 1
    
    #truncate unique tags by a certain threshold number
    print(f'{len(unique_tags_list)} unique tags before truncating...')
    for descriptor in unique_count.keys():
        if unique_count[descriptor] < args.min_unique_tags:
            unique_tags_list.remove(descriptor)
            print(f'Removing {descriptor} from tag list')
    print(f'{len(unique_tags_list)} unique tags after truncating...')
    
    with open(f'{args.save_dir}/unique_tags_list.json','w') as f:
        print(f'Dumping ordered tags list to {args.save_dir}/unique_tags_list.json')
        json.dump(unique_tags_list, f, indent=4)

    print(f'there are {len(unique_tags_list)} unique tags')
    print("Init as vectors of zeroes...")
    descriptor_matrix = descriptor_dict.copy()#copy without reference
    for uid in descriptor_matrix: #init matrix as vectors of zeroes of length the number of unique UMLS terms
        descriptor_matrix[uid] = [0]*len(unique_tags_list)

    #assign known classifications to the corresponding bit in each vector
    largest_condition_sum = 0
    print("Fill with knowns at corresponding locations...")
    for uid in descriptor_dict:
        if descriptor_dict[uid] != []:
            for descriptor,classification in descriptor_dict[uid]:
                # print(descriptor)
                if descriptor in unique_tags_list:#if we haven't deleted the descriptor due to it being too few
                    descriptor_matrix[uid][unique_tags_list.index(descriptor)] = classification
        if sum(descriptor_matrix[uid]) > largest_condition_sum:
            largest_condition_sum = sum(descriptor_matrix[uid])
    print(f'most dense target vector is {largest_condition_sum} dense')

    # print(descriptor_matrix)
    print("Mapping the uid to filenames...")
    # print(descriptor_matrix)
    with open(f'{args.save_dir}/matrix_dict_by_uid.json','w') as f:
        print(f'Dumping descriptor_matrix to {args.save_dir}/matrix_dict_by_uid.json')
        json.dump(descriptor_matrix, f, indent=4)

    map_dict = {}
    with open(args.map_file,'r') as f: #none of the default workflows ie from_csv, to_dict, etc worked well
        for line in f.readlines():
            linelist = line.strip().split(',')
            if linelist[0] == 'uid':
                continue
            if linelist[2] != 'Lateral':
                map_dict[linelist[0]] = linelist[1]
    # print(map_dict)

    img2vect = {}

    for uid in descriptor_matrix.keys():
        try: #avoiding when there is only lateral imaging
            # print(map_dict[uid])
            img2vect[map_dict[uid]] = descriptor_matrix[uid]
        except:
            with open(f'{args.save_dir}/.lateralsonly.txt','a') as f:
                f.write(f'lateral for uid {uid}.\n')

    with open(f'{args.save_dir}/img2vect.json','w') as f:
        print(f'Dumping img2vect to {args.save_dir}/img2vect.json')
        json.dump(img2vect, f, indent=4)
    
    return img2vect
    
def split_splits(img2vect, args):
    img2vect = list(img2vect.items())
    random.shuffle(img2vect)
    if type(args.split) != float:
        exit("Split val must be float")
    train_len = np.floor(len(img2vect)*args.split)
    print(f'Training size is {train_len}...')
    train_split = {}
    val_split = {}
    imgs_processed = 0
    for img, vect in img2vect:
        if imgs_processed < train_len:
            train_split[img] = vect
        else:
            val_split[img] = vect
        imgs_processed += 1
    print(len(train_split))
    print(len(val_split))

    if len(img2vect) != len(train_split)+len(val_split):
        exit('We lost an image somewhere!')

    with open(f'{args.save_dir}/train.json','w') as f:
        print(f'Dumping train_split to {args.save_dir}/train.json')
        json.dump(train_split, f, indent=4)
    with open(f'{args.save_dir}/valid.json','w') as f:
        print(f'Dumping val_split to {args.save_dir}/valid.json')
        json.dump(val_split, f, indent=4)
    
    print('Finished ripping from xml and converting to train/test split jsons. Have a nice day.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate the tags matrix from the cleaned xml.')
    parser.add_argument('--xml_dir', type=str, help='Where is the xml', required=True)
    parser.add_argument('--save_dir', help='Save the dict where?', required=True)
    parser.add_argument('--overwrite', help='Overwrite?', required=False, default=False)
    parser.add_argument('--map_file',help='Where is the file named Indiana projections or similar?')
    parser.add_argument('--split',type=float,help='What percent is training?',required=True)
    parser.add_argument('--min_unique_tags',type=int,help='Delete tags below this threshold.',required=True)
    args = parser.parse_args()
    if args.overwrite == False:
        print("I will not overwrite anything.")
    else:
        if os.path.exists(args.save_dir) == False:
            os.mkdir(args.save_dir)
        label_dict = load_xml(args.xml_dir)
        img2vect = matrix_from_tags(label_dict)
        split_splits(img2vect, args)