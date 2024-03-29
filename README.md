# Decision Transformer in JAX

## Features
1. Atari Env for evaluation uses `gymnasium`, instead of `atari_py`
2. Support `Tensorboard` & `WandB`
3. Better performance than offical coding in `Breakout`
4. Three models related to Transformer with offline RL: DT, RADT, StARformer (see [reference](#reference))

## Training Curve
Run 3 different seeds for each model, test 10 times at each evaluation. [WandB online](https://api.wandb.ai/links/wty-yy/06tql545)

![Training Curve in WandB](./asserts/Training%20Curve.png)

## GIF
Evaluate `Breakout` by model `StARformer` training with `seed=0`

<img src="./asserts/391_breakout.gif" alt="Breakout by StARformer" width="250"/>

Evaluate `Pong` by model `StARformer` training with `seed=0`

<img src="./asserts/20_pong.gif" alt="Breakout by StARformer" width="250"/>

## Start
### Download Dataset
Use the same training dataset as [GitHub-decision-transformer](https://github.com/kzl/decision-transformer),
download dataset, please refer to [readme-atari.md](https://github.com/kzl/decision-transformer/blob/master/atari/readme-atari.md). Since **we just need 1% dataset**, bellow cmd can download minimal size:
```shell
mkdir ./dqn_replay/Breakout
gsutil -m cp -R gs://atari-replay-datasets/dqn/Breakout/1/ ./dqn_replay/Breakout
```

### Training
- [`run.sh`](run.sh) for Decision Transformer.
- [`run_RADT.sh`](run_RADT.sh) for Return-Aligned Decision Transformer.
- [`run_StAR.sh`](run_StAR.sh) for StARformer.

Change the `--path-buffer-root` to your `dqn_replay` path in each running script.

## Reference

1. Decision-Transformer: https://arxiv.org/pdf/2106.01345.pdf, official code: https://github.com/kzl/decision-transformer
2. Return-Aligned Decision Transformer: https://arxiv.org/pdf/2402.03923.pdf
3. StARformer: Transformer with State-Action-Reward Representations for Visual Reinforcement Learning: https://arxiv.org/pdf/2110.06206.pdf, official code: https://github.com/elicassion/StARformer

