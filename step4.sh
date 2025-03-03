CUDA_VISIBLE_DEVICES=0 accelerate launch -m lm_eval --model hf \
    --model_args pretrained=save_LoRA_Null_adapter_llama2_PT_128_math_Null_v1_merged \
    --output_path result_path/result.json \
    --tasks  triviaqa,webqs,nq_open\
    --batch_size 64 \
    --max_batch_size 64 \
    --device cuda