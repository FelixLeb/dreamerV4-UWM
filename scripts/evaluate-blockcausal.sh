python scripts/eval_block_causal.py \
  --config-name dynamics/g1-large \
  --dynamics-ckpt /scratch/rk4342/projects/dreamerV4-UWM/checkpoints/dynamics/g1-may29-block-causal-sanity-4gpu-new/80000.pt \
  --tokenizer-ckpt checkpoints/tokenizer/g1.pt \
  --data-dir /scratch/rk4342/datasets/G1/wm --dataset-kind g1_chunked \
  --output-dir results/g1-block-causal-sanity-4gpu-new/80000 \
  --block-sizes 1,2,4,8,16 --num-context 8 --num-predict 48 --num-diffusion-steps 8 --num-samples 4

# python scripts/eval_block_causal.py \
#   --config-name dynamics/pushT-large \
#   --dynamics-ckpt /scratch/rk4342/projects/dreamerV4-UWM/checkpoints/dynamics/pushT-block-causal-sanity-4gpu-new/97500.pt \
#   --tokenizer-ckpt checkpoints/tokenizer/pushT.pt \
#   --data-dir /scratch/rk4342/datasets/pushT/play --dataset-kind sharded_hdf5 \
#   --output-dir results/pushT-block-causal-sanity-4gpu-new/97500 \
#   --block-sizes 1,2,4,8,16 --num-context 8 --num-predict 48 --num-diffusion-steps 8 --num-samples 4