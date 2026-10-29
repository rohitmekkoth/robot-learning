# Week 1 baselines

1. Eval pretrained checkpoint (Mac):
   lerobot-eval --policy.path=lerobot/diffusion_pusht --env.type=pusht --eval.n_episodes=50 --policy.device=mps
2. Train Diffusion Policy on PushT (GPU, after approval).
3. Train ACT on ALOHA insertion (GPU, after approval).

Record every result in results.csv.
