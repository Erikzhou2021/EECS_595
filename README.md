# Prompting for Fairness: Mitigating Gender Bias in Large Language Models with Debias Prompting

## In StereoSet_CrowS folder:

### 1) Install dependencies in new conda environment 
```
conda create -n myenv python=3.9
conda activate myenv
cd bias-bench 
python -m pip install -e .
pip install pandas==1.5.3
```
### 2) Run CrowS-Pairs debias prompts
Examples below test GPT2 model gender bias on CrowS-Pairs bias benchmark\
**Baseline no prompt:** `python experiments/crows.py --model GPT2LMHeadModel --model_name_or_path gpt2 --bias_type gender`\
**Debias prompts:** `python experiments/crows_debias.py --model SelfDebiasGPT2LMHeadModel --model_name_or_path gpt2 --bias_type gender`

### 3) Run StereoSet debias prompts
Examples below test GPT2 model gender bias on StereoSet bias benchmark\
**Baseline no prompt:**
```
python experiments/stereoset.py --model GPT2LMHeadModel --model_name_or_path gpt2 --bias_type gender --seed 1
# Run stereoset_evaluation.py on the json file, look at arguments
python experiments/stereoset_evaluation.py --predictions_file predictions_file_name --prediction_dir folder_name
```
**Debias prompts:**
```
python experiments/stereoset.py --model SelfDebiasGPT2LMHeadModel --model_name_or_path gpt2 --bias_type gender --seed 1
# Run stereoset_evaluation.py on the json file, look at arguments
python experiments/stereoset_evaluation.py --predictions_file predictions_file_name --prediction_dir folder_name
```
## In BBQ folder:
### 1) GreatLakes: Install dependencies in new conda environment 
**requirements1.txt**
```
huggingface-hub==0.17.3
numpy==1.26.1
pandas==2.1.1
protobuf==4.25.2
sentencepiece==0.1.99
```

**requirements2.txt**
```
tokenizers==0.14.1
tqdm==4.66.1
transformers==4.34.0
```

```
module load python3.10-anaconda/2023.03
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements1.txt
pip install -r requirements2.txt
```

### 2) Run the following bash files to get bias metric results
* run_mistral.sh
* run_mistral_mine.sh
* run_llama.sh
* run_gpt2-xl.sh
* run_opt-1.3b.sh
