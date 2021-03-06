#!/usr/bin/env bash
cd ..
python3 train.py --model_dir model/sample\
 --batch_size 10\
 --save_freq 50\
 --valid_freq 5 --source_vocabulary dataset/summerization_sample/vocab.json --target_vocabulary dataset/summerization_sample/vocab.json --source_train_data dataset/summerization_sample/articles.train.sample.txt --target_train_data dataset/summerization_sample/summaries.train.sample.txt --source_valid_data dataset/summerization_sample/articles.eval.sample.txt --target_valid_data dataset/summerization_sample/summaries.eval.sample.txt --num_encoder_symbols 21548 --num_decoder_symbols 21548