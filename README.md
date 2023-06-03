# VQA-Rad-with-BERT
BERT-version BAN on VQA-Rad 

## Notices
- Many, including the config 

## Steps for training 

### Example: run `configs/qcr_pubmedclipRN50_ae_rad_nondeterministic.yaml`
- Activate your virtual env. 
- Copy the essentials to this folder from PubMedCLIP/QCR_PubMedCLIP, including datasets, caches, image pickles and so on. 
  - Be aware not to overwrite the files in this folder. 
  - The below files (as far as I can remember) are heavily rewritten by @Nana2929, and are the core parts of this BERT revision. 
  - `lib/BAN/multi_level_model.py`
  - `lib/config/default.py`
  - `lib/language`
  - `main.py`
- Download the new checkpoint for type_classifier. Link: [to be updated](). 
  - The BERT model used now is `emilyalsentzer/Bio_ClinicalBERT`. Make sure the checkpoint has `bio` in its name.
  - Make sure `main.py` #L99 loads this checkpoint. 
- Install compatible torch version. 
- Run `python3 main.py --cfg=configs/qcr_pubmedclipRN50_ae_rad_nondeterministic.yaml` 

## Testing
還沒寫，說不定根本用不到那裡 QQ。
