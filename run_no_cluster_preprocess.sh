#!/bin/bash

if [[ $# < 2 ]]; then
    echo "Usage: $0 \${DATASET} \${LABEL_EMB}"
    exit -1
fi

DATASET=$1
LABEL_EMB=$2    # pifa-tfidf | pifa-neural | text-emb

DATA_DIR=datasets

OUTPUT_DIR=no_cluster_models/${DATASET}
PROC_DATA_DIR=${OUTPUT_DIR}/proc_data
mkdir -p ${PROC_DATA_DIR}

LABEL_EMB_NAME=${LABEL_EMB}
INDEXER_DIR=${OUTPUT_DIR}/${LABEL_EMB_NAME}/indexer
python -u -m xbert.preprocess \
    --do_proc_label \
    -i ${DATA_DIR}/${DATASET} \
    -o ${PROC_DATA_DIR} \
    -l ${LABEL_EMB_NAME} \
    -c ${OUTPUT_DIR}/code-1to1.npz

echo "Output to dir ${PROC_DATA_DIR}"
