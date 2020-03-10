#!/usr/bin/env bash

EXEC_ID=Robera-base-masker-32768-p512-b256-c0.3
DATA_DIR=../data-bin/wiki_book_32768
TOTAL_UPDATES=1000000
WARMUP_UPDATES=10000
PEAK_LR=0.0001
TOKENS_PER_SAMPLE=512
MAX_POSITIONS=512
MAX_SENTENCES=8 # 32 for v100 and fp16
UPDATE_FREQ=4
SEED=100
DIR=../exps/masker_3.10_log/average_lambda_1e-6

echo 'Environment'
nvidia-smi
#ls -alh
#ls ~ -alh


echo 'Start Training'
python train.py --fp16 ${DATA_DIR} --num-workers 4  --ddp-backend=no_c10d   \
    --task masked_lm --criterion mask_co_leaner --masker_lambda 1e-6 \
    --masker_m -1.0 --masker_eps 0.2 \
    --arch roberta_leaner --sample-break-mode complete --tokens-per-sample ${TOKENS_PER_SAMPLE} \
    --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr ${PEAK_LR} --warmup-updates ${WARMUP_UPDATES} --total-num-update ${TOTAL_UPDATES} \
    --dropout 0.1 --attention-dropout 0.1 --activation-dropout 0.1 --weight-decay 0.01 \
    --max-sentences ${MAX_SENTENCES} --update-freq ${UPDATE_FREQ} --seed ${SEED} \
    --encoder-normalize-before  \
    --max-update ${TOTAL_UPDATES} --log-format simple --log-interval 500 --tensorboard-logdir ${DIR} \
    --distributed-world-size 8 --distributed-rank 4 --distributed-init-method "tcp://10.0.13.6:8080" \
    --save-interval-updates 50000 --keep-interval-updates 6 --no-epoch-checkpoints --skip-invalid-size-inputs-valid-test --save-dir ${DIR}/model
