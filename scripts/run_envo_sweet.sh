#!/bin/bash
set -e

kroma run \
    --method_config=kroma_scibert_envo_sweet \
    --llm=meta-llama/Llama-3.3-70B-Instruct-Turbo-Free \
    --baseline
