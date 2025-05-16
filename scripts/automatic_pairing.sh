#!/bin/bash

# Default values
PRETRAIN_MODEL_NAME=""
CKPT=""
CROSS_EMBODIMENT=""
NUM_CHOPS=4
LOOKUP_TYPE="ot"  # Default to ot, can be "ot", "tcc", or "both"

# Function to display usage
usage() {
    echo "Usage: $0 --pretrain_model_name <name> --checkpoint <num> --cross_embodiment <type> [--num_chops <num>] [--lookup_type <type>]"
    echo "  --pretrain_model_name: Pretrain model name (required)"
    echo "  --checkpoint: Checkpoint number (required)"
    echo "  --cross_embodiment: Cross embodiment (sphere-easy, sphere-medium, sphere-hard) (required)"
    echo "  --num_chops: Number of clips to retrieve per robot video (default: 4)"
    echo "  --lookup_type: Lookup type (ot, tcc) (default: ot)"
    exit 1
}

# Parse command line arguments
OPTS=$(getopt -o '' --long pretrain_model_name:,checkpoint:,cross_embodiment:,num_chops:,lookup_type: -n "$0" -- "$@")
if [ $? != 0 ] ; then echo "Failed to parse options" >&2 ; usage ; fi
eval set -- "$OPTS"

# Extract options and their arguments
while true; do
    case "$1" in
        --pretrain_model_name ) PRETRAIN_MODEL_NAME="$2"; shift 2 ;;
        --checkpoint ) CKPT="$2"; shift 2 ;;
        --cross_embodiment ) CROSS_EMBODIMENT="$2"; shift 2 ;;
        --num_chops ) NUM_CHOPS="$2"; shift 2 ;;
        --lookup_type ) LOOKUP_TYPE="$2"; shift 2 ;;
        -- ) shift; break ;;
        * ) break ;;
    esac
done

# Check required parameters
if [ -z "$PRETRAIN_MODEL_NAME" ] || [ -z "$CKPT" ] || [ -z "$CROSS_EMBODIMENT" ]; then
    echo "Missing required parameters!"
    usage
fi

# Validate lookup type
if [ "$LOOKUP_TYPE" != "ot" ] && [ "$LOOKUP_TYPE" != "tcc" ]; then
    echo "Invalid lookup type. Must be 'ot' or 'tcc'."
    usage
fi

echo "======================================================================"
echo "Starting automatic pairing of cross-embodiment datasets..."
echo "Using pretrain model: $PRETRAIN_MODEL_NAME"
echo "Checkpoint: $CKPT"
echo "Cross embodiment: $CROSS_EMBODIMENT"
echo "Number of chops: $NUM_CHOPS"
echo "Lookup type: $LOOKUP_TYPE"
echo "======================================================================"

# Step 2a: Convert images into latent vectors using pretrained visual encoder
echo -e "\n>> Step 2a: Converting images into latent vectors..."
python scripts/label_sim_kitchen_dataset.py \
    pretrain_model_name=$PRETRAIN_MODEL_NAME \
    ckpt=$CKPT \
    cross_embodiment=$CROSS_EMBODIMENT

if [ $? -ne 0 ]; then
    echo "Error in Step 2a. Exiting."
fi

# Step 2b: Compute and store sequence-level distance metrics
echo -e "\n>> Step 2b: Computing sequence-level distance metrics..."
python scripts/chopped_segment_wise_dists.py \
    pretrain_model_name=$PRETRAIN_MODEL_NAME \
    ckpt=$CKPT \
    num_chops=$NUM_CHOPS \
    cross_embodiment=$CROSS_EMBODIMENT

if [ $? -ne 0 ]; then
    echo "Error in Step 2b. Exiting."
fi

# Step 2c: "Imagine" the paired demonstrator dataset
echo -e "\n>> Step 2c: Imagining the paired demonstrator dataset..."
if [ "$LOOKUP_TYPE" = "ot" ]; then
    python scripts/reconstruction.py \
        pretrain_model_name=$PRETRAIN_MODEL_NAME \
        ckpt=$CKPT \
        num_chops=$NUM_CHOPS \
        cross_embodiment=$CROSS_EMBODIMENT \
        ot_lookup=True \
        tcc_lookup=False
else  # tcc
    python scripts/reconstruction.py \
        pretrain_model_name=$PRETRAIN_MODEL_NAME \
        ckpt=$CKPT \
        num_chops=$NUM_CHOPS \
        cross_embodiment=$CROSS_EMBODIMENT \
        ot_lookup=False \
        tcc_lookup=True
fi

if [ $? -ne 0 ]; then
    echo "Error in Step 2c. Exiting."
fi

# Step 2d: Convert the imagined dataset into latent vectors
echo -e "\n>> Step 2d: Converting the imagined dataset into latent vectors..."
# Determine the imagined dataset name based on lookup type
if [ "$LOOKUP_TYPE" = "ot" ]; then
    IMAGINED_DATASET="${PRETRAIN_MODEL_NAME}_${CROSS_EMBODIMENT}_generated_ot_${NUM_CHOPS}_ckpt${CKPT}"
else
    IMAGINED_DATASET="${PRETRAIN_MODEL_NAME}_${CROSS_EMBODIMENT}_generated_tcc_${NUM_CHOPS}_ckpt${CKPT}"
fi

python scripts/label_retrieved_dataset.py \
    pretrain_model_name=$PRETRAIN_MODEL_NAME \
    ckpt=$CKPT \
    imagined_dataset=$IMAGINED_DATASET

if [ $? -ne 0 ]; then
    echo "Error in Step 2d. Exiting."
fi

echo -e "\n======================================================================"
echo "Imagined dataset generated: $IMAGINED_DATASET"
echo "======================================================================"