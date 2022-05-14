# granular-tags
Granular tag prediction for IU-x-ray

To pull github and set up environment run in your home directory (prerequisites: anaconda and an internet connection)
```shell
conda create -n grantags python=3.7 --yes
conda activate grantags
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch --yes
pip install kaggle
pip install gdown
mkdir .kaggle
cd .kaggle
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1zo5krOvThWMfO-8qV4EmIOoZL_dCTSD1' -O kaggle.json
chmod 600 kaggle.json
cd
git clone https://github.com/Kameradna/granular-tags.git
cd granular-tags
kaggle datasets download -d raddar/chest-xrays-indiana-university
mkdir IU_xray_data
unzip chest-xrays-indiana-university.zip -d IU_xray_data
gdown 'https://docs.google.com/uc?id=19nAgOOGf6WK57xqqmPy9dIvOoupHcwsP' -O reports_and_files.zip
unzip reports_and_files.zip
export THESISPATH="$PWD" #record parent path for use in arguments
cd big_transfer
```

Now choose which network you would like and download it into big_transfer
```shell
wget https://storage.googleapis.com/bit_models/BiT-M-R50x1.npz
```
See https://github.com/google-research/big_transfer.git for the Google Big Transfer documentation, but the choices here are:
ResNet-50x1, ResNet-101x1, ResNet-50x3, ResNet-101x3, and ResNet-152x4 for BiT-M (ImageNet21k) and BiT-S (ImageNet1k).


And run the model training (inside the big_transfer directory)
```shell
python -m bit_pytorch.train --name testingiuxray_`date +%F_%H%M%S` --model BiT-M-R50x1 --logdir $THESISPATH/output/testing --dataset iu-xray --datadir $THESISPATH/IU_xray_data/images/images_normalized --workers 8 --batch 128 --batch_split 4 --eval_every 10 --base_lr 0.001 --annodir $THESISPATH/splits
```
As it has been set up, every time you restart the shell, you should reinitialise the conda environment and add the environment variable of the parent directory of the git repo.
```shell
conda activate grantags
cd granular-tags
export THESISPATH="$PWD" #record parent path for use in arguments
cd big_transfer
```
Below this line is for more custom use cases, and requires further dependencies such as bioc, others


To rerun the matrix_from_tags section, which transforms the negbio2 processed and classified final clean xml file into the target vectors and image mappings:
```shell
python matrix_from_tags.py --xml_dir=$PWD/xml_reports/clean/all_together_all_topics.secsplit.ssplit.bllip.ud.mm.neg2.negbio.xml --save_dir=splits3 --overwrite=True --map_file=$PWD/IU_xray_data/indiana_projections.csv --split 0.9 --min_unique_tags 10
```
To rerun the full negbio pipeline would be very difficult and classically I have run into many dependency issues, but should you desire to do so, return to granular-tags directory and run:
```shell
git clone https://github.com/bionlplab/negbio2.git
conda create --name negbio2 python=3.7 pip pandas
conda activate negbio2
export OUTPUT_DIR="$PWD/xml_reports"
export TEXT_DIR="$PWD/txt_reports"
export NEGBIOC_DIR="$PWD/negbio2"
export PYTHONPATH="$PWD/negbio2"
cd negbio2
pip install -r requirements3.txt
cd ..
mkdir $OUTPUT_DIR #just in case
python negbio2/negbio/negbio_pipeline.py text2bioc --output=$OUTPUT_DIR/all_together_find_ind.xml $TEXT_DIR/*.txt
python negbio2/negbio/negbio_pipeline.py section_split --output=$OUTPUT_DIR/section_split $OUTPUT_DIR/all_together_find_ind.xml
python negbio2/negbio/negbio_pipeline.py ssplit --output $OUTPUT_DIR/ssplit $OUTPUT_DIR/section_split/* --workers=8
python negbio2/negbio/negbio_pipeline.py parse --output $OUTPUT_DIR/parse $OUTPUT_DIR/ssplit/* --workers=4
python negbio2/negbio/negbio_pipeline.py ptb2ud --output $OUTPUT_DIR/ud $OUTPUT_DIR/parse/* --workers=8
python negbio2/negbio/negbio_pipeline.py dner_regex --phrases $NEGBIOC_DIR/patterns/chexpert_phrases.yml --output $OUTPUT_DIR/dner $OUTPUT_DIR/ud/* --suffix=.chexpert-regex.xml --workers=6
python negbio2/negbio/negbio_pipeline.py neg2 --neg-patterns=$NEGBIOC_DIR/patterns/neg_patterns2.yml --pre-negation-uncertainty-patterns=$PWD/negbio2/patterns/chexpert_pre_negation_uncertainty.yml --post-negation-uncertainty-patterns=$PWD/negbio2/patterns/post_negation_uncertainty.yml --neg-regex-patterns=$PWD/negbio2/patterns/neg_regex_patterns.yml --uncertainty-regex-patterns=$NEGBIOC_DIR/patterns/uncertainty_regex_patterns.yml --workers=6 --output $OUTPUT_DIR/neg $OUTPUT_DIR/dner/*
python negbio2/negbio/negbio_pipeline.py cleanup --output $OUTPUT_DIR/clean $OUTPUT_DIR/neg/*
python matrix_from_tags.py --xml_dir=$OUTPUT_DIR/clean/all_together_all_topics.secsplit.ssplit.bllip.ud.chexpert-regex.neg2.negbio.xml --save_dir=splitsX --overwrite=True --map_file=$PWD/IU_xray_data/indiana_projections.csv --split 0.8 --min_unique_tags 0
```
Consider providing alternate names for your new xml_reports and splits folders to prevent overwrites, and I would highly recommend running at least the negbio pipeline one command at a time to ensure that errors are not propagated. Be advised that some steps ie the parse or pt2ud steps take up to 40 or more minutes depending on performance.


You may want to edit the xmlify.py (name tbc) file and rerun to repopulate the txt_reports folder with text reports that include or exclude additional fields available in the OpenI data. Canonically we just base models on Impression and Findings since these are the most generic 'report' type data that is available, and makes the results we get most useful across different datasets since many do not include any other fields of information.


Also consider that you have to install the MetaMap binaries in their default location and specify that when using the MetaMap tagger. Eg: with metamap2020 installed (you need to apply for a license and wait a couple of days to get access to the download and API).

```shell
export METAMAP_BIN=METAMAP_HOME/bin/metamap20
python negbio2/negbio/negbio_pipeline.py dner_mm --metamap=$METAMAP_BIN --output $OUTPUT_DIR/dner $OUTPUT_DIR/ud/* --suffix=.chexpert-regex.xml --workers=8
```
