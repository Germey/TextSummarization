#!/usr/bin/env bash
cd ..
python3 decode.py --gpu 2 --model_path model/bpe_without_date/summary.ckpt-30100 --decode_input dataset/nlpcc_bpe_without_date/articles.test.txt --decode_output dataset/nlpcc_bpe_without_date/summaries.test.txt