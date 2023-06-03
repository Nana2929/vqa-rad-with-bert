# VQA-Rad-with-BERT
BERT-version BAN on VQA-Rad

## Notices
- Many, including the config

## Steps for training
(Updated: 2023/06/03 23:30)
- [Link to type_classifier_rad_biobert_2023Jun03-155924.pth](https://drive.google.com/file/d/1Y958ocG58HM52ZFQsbbKrjiMt3G5u5nz/view?usp=sharing).
  - The BERT model used now is `emilyalsentzer/Bio_ClinicalBERT`. Make sure the checkpoint has `bio` in its name.
- config name: `qcr_pubmedclipRN50_ae_rad_nondeterministic_2lrs.yaml`


### Example: run `configs/{cfg_name}`
- Activate your virtual env.
- Copy the essentials to this folder from PubMedCLIP/QCR_PubMedCLIP, including datasets, caches, image pickles and so on.
  - Be aware not to overwrite the files in this folder.
  - The below files (as far as I can remember) are heavily rewritten by @Nana2929, and are the core parts of this BERT revision.
  - `lib/BAN/multi_level_model.py`
  - `lib/config/default.py`
  - `lib/language`
  - `main.py`
  - ...
- Download the new checkpoint for type_classifier (see above).
  - Make sure `main.py` #L99 loads this checkpoint.
- Install compatible torch version.
- Run `python3 main.py --cfg=configs/{cfg_name}`

## Testing
還沒寫，說不定根本用不到那裡 QQ。
