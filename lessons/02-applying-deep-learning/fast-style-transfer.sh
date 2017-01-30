#!/bin/sh
cd fast-style-transfer
for checkpoint in ../checkpoints/*; do
  echo "Styling with $checkpoint"
  python evaluate.py --checkpoint $checkpoint --in-path ../duke.jpg --out-path ../duke-$(basename ${checkpoint} .ckpt).jpg
  echo "Done styling with $checkpoint"
done
