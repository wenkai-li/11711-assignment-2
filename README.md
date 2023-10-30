# 11711-assignment-2

# Setup

```bash
git clone https://github.com/wenkai-li/11711-assignment-2.git
cd 11711-assignment-2
conda create -n assignment2 python=3.9
conda activate assignment2
pip install -r requirements.txt
```

# Continue Pretrain

```bash
python roberta_continue_pretrain.py \
--model_name_or_path roberta-base  \
--config_name roberta-base \
--tokenizer_name roberta-base  \
--line_by_line \
--save_total_limit 2 \
--load_best_model_at_end true \
--evaluation_strategy steps \
--save_strategy steps \
--no_use_fast_tokenizer \
--validation_split_percentage 10 \
--do_train \
--do_eval \
--per_device_train_batch_size 32 \
--per_device_eval_batch_size 16 \
--learning_rate 1e-5 \
--max_seq_length 512 \
--num_train_epochs 200
```

# Finetune

```bash
python3 froberta_finetuning.py
```

# Predict

```bash
python3 roberta_predict.py
```

# LLM

```bash
python3 llama2.py
```

