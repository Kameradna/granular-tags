# granular-tags
Granular tag prediction for IU-x-ray

To pull github and set up environment run in your home directory
```shell
conda create -n grantags python=3.7 --yes
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch --yes
conda activate grantags
cd .kaggle
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1zo5krOvThWMfO-8qV4EmIOoZL_dCTSD1' -O kaggle.json
cd
git clone https://github.com/Kameradna/granular-tags.git
cd granular-tags
kaggle datasets download -d raddar/chest-xrays-indiana-university
unzip chest-xrays-indiana-university.zip
gdown 'https://docs.google.com/uc?id=19nAgOOGf6WK57xqqmPy9dIvOoupHcwsP' -O reports_and_files.zip
unzip reports_and_files.zip
export THESISPATH="$PWD" #record parent path for use in arguments
cd big_transfer
```

Now choose which network you would like and download it into big_transfer
```shell
wget https://storage.googleapis.com/bit_models/BiT-M-R50x1.npz
```
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
Definitely consider changing the training validation split via matrix_to_tags
```shell
python matrix_from_tags.py --xml_dir=$PWD/xml_reports/clean/all_together_all_topics.secsplit.ssplit.bllip.ud.mm.neg2.negbio.xml --save_dir=splits3 --overwrite=True --map_file=$PWD/IU_xray_data/indiana_projections.csv --split 0.9 --min_unique_tags 10
```
