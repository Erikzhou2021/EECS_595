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
Examples below testing GPT2 model gender bias on CrowS-Pairs bias benchmark\
**Baseline no prompt:** `python experiments/crows.py --model GPT2LMHeadModel --model_name_or_path gpt2 --bias_type gender`\
**Debias prompts:** `python experiments/crows_debias.py --model SelfDebiasGPT2LMHeadModel --model_name_or_path gpt2 --bias_type gender`

