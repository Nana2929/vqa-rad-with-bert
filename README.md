# VQA-Rad with 🤗 BERT 
- BERT-version Bilinear Attn Networks on VQA-Rad
- ⚠️ Very quick revision (done in 5 days 😅) so the overall code structure may look ugly. Thank you for your understanding and if you find any bugs, make a PR or open an issue. 
## Model Architecture 
1. Bilinear Attn Networks: BERT-version
    - Downstream tasks: VQA-Rad 
    - Revised based on [sarahESL/PubMedCLIP (2021)](https://github.com/sarahESL/PubMedCLIP/tree/main/QCR_PubMedCLIP/main) 
    - Explanation: Original BAN model uses normal `nn.Embeddings` initialized with glove 300d and GRU as text encoder.  
    <img width="756" alt="image" src="https://github.com/Nana2929/vqa-rad-with-bert/assets/58811089/f9c9959d-f98c-457d-a475-84b34f535d84">
2. Use pretrained Bio-Clinical BERT.  
3. Train with 2 optimizers, because BAN and BERT require very different learning rates. 

## Performance 
- Experiment stats show that pretrained CLIP visual encoder `RN50x4` with our BERT-BAN and the preprocessed image outperforms the original PubMedCLIP ($71.62\% \rightarrow 73.17\%$). With original images, it achieves $72.28\%$. Note that the $71.62\%$ is our reproduced score of paper instead of paper's score ($71.8\%$). 
- For more details, see [2023 MIS: Final Presentation Slides](https://docs.google.com/presentation/d/1XeD1r8T_veCGpU5ApmqVB3gS1tnZ1ssIjdcxCGCUEtk/edit?usp=sharing).
- ⚠️ It is likely that some settings could still be changed to make the performance better.

## Running Experiments
### Download Data
   - From [`Awenbocc/med-vqa/data`](https://github.com/Awenbocc/med-vqa/tree/master/data) you can find the `images` and img pickles. 
   - If you'd like to pickle the data from `images` on your own:
       - Open `lib/utils/run.sh`. 
       - Configure the `IMAGEPATH`.
       - Run the`create_resized_images.py` lines to put the new image pickles under`DATARADPATH`. 
       - The VQA script reads the image pickles from your`DATARADPATH` so be sure they are placed correctly. 
### Prepare an Answer-type Classifier (closed/open)
- This classifier is used in validation period, where a question is classified into Open or Close, and then sent to different answer pools for the 2nd stage answer classififcation. 
- Please download and unzip [`type_classifier_rad_biobert_2023Jun03-155924.pth.zip`](https://drive.google.com/file/d/1-mCz91DxzdA0kVd78MTM5Blvfdyu3uyc/view?usp=sharing) for a pretrained type classifier. The BERT model for this type classifier checkpoint is `emilyalsentzer/Bio_ClinicalBERT`. 
- If the type classifier is corrupted (it seems that uploading it anywhere corrupts it, only `scp` resolves the issue), run `type_classifier.py` in the repo again to train a new one. 

- ⚠️ The config passed should be the one you will be using in the VQA training. Specifically, make sure the config variable `DATASET/EMBEDDER_MODEL` is consistent with the following experiments' config so that their vocab sizes match. 
- ⚠️ If you'd like to try out other BERT-based models, feel free to change config variable `DATASET/EMBEDDER_MODEL` to another huggingface model name, and then train and use your own type classifier.  

### Run Training 
   - Create a virtual env and then `pip install -r requirements.txt`. 
   - Install `torch` series packages following [start locally|Pytorch](https://pytorch.org/get-started/locally/). 
   - Open a config that you'd like to use and check:
       - For `TRAIN.VISION.CLIP_PATH`, download the pretrained clip visual encoders [here](https://onedrive.live.com/redir?resid=132993BDA73EE095!384&authkey=!APg2nf5_s4MCi3w&e=zLVlJ2). Read [`SarahESL/PubMedCLIP/PubMedCLIP/README.md`](https://github.com/sarahESL/PubMedCLIP/tree/main/PubMedCLIP) formore details.
       - Change `DATASET.DATA_DIR` to your dataset's path.  
   - Copy the essentials to this folder from [`SarahESL/PubMedCLIP/QCR_PubMedCLIP`](https://github.com/sarahESL/PubMedCLIP/tree/main/QCR_PubMedCLIP) if anything is missing.  
   - Run `python3 main.py --cfg={config_path}`
### Notes
- Be sure to use modified configs, namely `configs/qcr_pubmedclip{visual_encoder_name}_ae_rad_nondeterministic_typeatt_2lrs.yaml`. 
- The changed files from BAN to BERT-BAN are:
    - `configs/`
    - `lib/config/default.py`
    - `lib/BAN/multi_level_model.py`
    - `lib/lngauge/classify_question.py`
    - `lib/lngauge/language_model.py`
    - `lib/dataset/dataset_RAD_bert.py`
    - (May be more)
- Beware of your disk space because 1 model checkpoint is roughly 3.6 GB; once your disk space is full the training stops. 

### Testing (Unsupported now)
We haven't written the test script (supposed to be used for creating the validation file). ``main/test.py`` is used for testing in original repo, so you could modify the eval loop by following `main/train.py`, which should be workable. 

## Extra  
Make a PR or open an issue for your questions and we may (or may not) deal with it if we find time.



