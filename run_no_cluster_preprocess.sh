#!/bin/bash
#
# Sample commandline:
# ./run_no_cluster_preprocess.sh Eurlex-4K roberta

if [[ $# < 2 ]]; then
    echo "Usage: $0 \${DATASET} \${MODEL_TYPE}"
    exit -1
fi

DATASET=$1
MODEL_TYPE=$2
MAX_XSEQ_LEN=128

# HuggingFace pretrained model preprocess
if [ $MODEL_TYPE == "bert" ]; then
    MODEL_NAME="bert-large-cased-whole-word-masking"
elif [ $MODEL_TYPE == "roberta" ]; then
    MODEL_NAME="roberta-large"
elif [ $MODEL_TYPE == 'xlnet' ]; then
    MODEL_NAME="xlnet-large-cased"
else
    echo "Unknown MODEL_NAME!"
    exit
fi

DATA_DIR=datasets

OUTPUT_DIR=no_cluster_models/${DATASET}
PROC_DATA_DIR=${OUTPUT_DIR}/proc_data
mkdir -p ${PROC_DATA_DIR}

python -u -m xbert.preprocess \
    --do_proc_label \
    -i ${DATA_DIR}/${DATASET} \
    -o ${PROC_DATA_DIR} \
    -c ${OUTPUT_DIR}/code-1to1.npz

mv ${PROC_DATA_DIR}/C.trn.*.npz ${PROC_DATA_DIR}/C.trn.npz
mv ${PROC_DATA_DIR}/C.tst.*.npz ${PROC_DATA_DIR}/C.tst.npz

python -u -m xbert.preprocess \
    --do_proc_feat \
    -i ${DATA_DIR}/${DATASET} \
    -o ${PROC_DATA_DIR} \
    -m ${MODEL_TYPE} \
    -n ${MODEL_NAME} \
    --max_xseq_len ${MAX_XSEQ_LEN} \
    |& tee ${PROC_DATA_DIR}/log.${MODEL_TYPE}.${MAX_XSEQ_LEN}.txt

echo "Output to dir ${PROC_DATA_DIR}"
