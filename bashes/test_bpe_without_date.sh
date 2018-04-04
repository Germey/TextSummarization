#!/usr/bin/env bash
cd ..
python3 decode.py --gpu 2 --batch_size 64 --source_max_length 1500 --target_max_length 60 --display_freq 5 --save_freq 100 --valid_freq 100 --model_path model/bpe_without_date/summary.ckpt-30100 --decode_input dataset/nlpcc_bpe_without_date/articles.test.txt --decode_output dataset/nlpcc_bpe_without_date/summaries.test.txt