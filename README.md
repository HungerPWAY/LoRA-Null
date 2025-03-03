# LoRA-Null: Low-Rank Adaptation via Null Space for Large Language Models

## Getting Start

Download the repo and install dependencies. 

```
cd LoRA_Null
pip install -r requirements.txt
```



## Step 1: 
sh step1.sh



```bash
CUDA_VISIBLE_DEVICES=0 python build_corda.py \
    --model_id "meta-llama/Llama-2-7b-hf" \
    --singular_aware \
    --r {rank} \
    --use_cache \
    --calib_dataset "nqopen" \
    --calib_loader_size 256 \
    --save_model \
    --save_path {path_to_decomposed_model}
```

**Arguments**:

- `--model_id` is the pre-trained model for decomposition.
- `--cov_aware` adopts our context-oriented decomposition and collects covariance matrices.
- `--r` is the low rank of LoRA, e.g. 128.
- `--use_cache` adopts the dataloader and covariance matrices saved in `Adapter/cache`, to avoid calculating the covariance matrices again.
- `--calib_dataset` specifies the dataset to sample data to obtain covariance matrices. We use QA datasets `"nqopen"`.
- `--calib_loader_size` is the number of sampled data. 
- `--save_model` saves the initialized model in `--save_path`. 


## Step 2: Adapter Training
sh step2.sh



## Step 3: Merging

After training, LoRA adapter can be merged with the base model by runing:
sh step3.sh


## Step 4: Inference on world knowledge**:

Inference on world knowledge benchmarks is based on [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). For example, we evaluate by:
sh step4.sh

```bash
accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained={path_to_merged_model},trust_remote_code=True,dtype=float16 \
    --output_path {result_path}/nq_open.json \
    --tasks nq_open,triviaqa,nq_open \
    --batch_size auto \
    --max_batch_size 8 \
    --device cuda
```


## Step 5: Inference on Downstream Tasks
**Inference on Math**:

Evaluation on Gsm8k and Math can be performed by:
sh step5.sh
```
sh tools/inference_Math.sh {path_to_merged_model}
```

**Inference on Code and Instruction Following**:

Evaluation on HumanEval and MBPP is based on [bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness). Evaluation on MTBench is based on [FastChat](https://github.com/lm-sys/FastChat). We use their default settings for evaluation. 

