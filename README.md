# VGAU-Diag

> 🎉 Accepted to **EMNLP 2026 Main Conference**.

Official implementation of **When Does Visual Generation Help Visual Understanding in Unified Multimodal Models?**

[[Paper](https://arxiv.org/abs/2608.22174)]

VGAU-Diag is a difficulty-aware evaluation framework for diagnosing visual generation-assisted understanding in unified multimodal models. It evaluates multiple reasoning paradigms under matched settings and uses oracle-assisted protocols to separate visual-generation failures from visual-understanding failures.

## Highlights

- Six executable visual-planning tasks: Maze, Sokoban, Path, Onet, Parking, and Klotski.
- Three difficulty levels with 100 instances per task.
- Direct, Text-CoT, Generate-then-Answer, Vision-CoT, and Visual-Reasoning protocols.
- Oracle visual aids for controlled generation–understanding diagnosis.
- Solver-based task-level evaluation with feasible and optimal-solution metrics.

## Installation

Python 3.10–3.12 is supported.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For Weights & Biases logging:

```bash
pip install -e ".[tracking]"
```

## BAGEL Setup

Clone the official [BAGEL repository](https://github.com/ByteDance-Seed/Bagel) and download its model weights. Then expose the checkout through `BAGEL_REPO`:

```bash
export BAGEL_REPO=/path/to/Bagel
```

Pass the downloaded weight directory through `pretrained` when running evaluation.

## Evaluation Protocols

Each task family provides the following variants:

- `*_direct`: direct answering.
- `*_cot`: textual chain-of-thought.
- `*_aux`: Oracle-GtA with a deterministic annotated visual aid.
- `*_gta`: Self-GtA using a model-generated visual aid.
- `*-step-by-step`: Vision-CoT; set `image_source=gt` for Oracle-VCoT or `image_source=model` for Self-VCoT.
- `*_aux_step-by-step`: augmented Vision-CoT with grid and coordinate cues.
- `*_vision_reasoning`: Self-VR using a generated visual solution.

Available task families are `maze`, `sokoban`, `path`, `onet`, `parking`, and `klotski`.

List every registered task:

```bash
vgau-eval --tasks list
```

Run Text-CoT:

```bash
vgau-eval \
  --model bagel \
  --model_args pretrained=/path/to/BAGEL-7B-MoT,mode=understanding \
  --tasks maze_cot \
  --batch_size 1 \
  --output_path outputs/maze_cot \
  --log_samples
```

Run Self-GtA:

```bash
vgau-eval \
  --model bagel \
  --model_args pretrained=/path/to/BAGEL-7B-MoT,mode=generation \
  --tasks maze_gta \
  --batch_size 1 \
  --output_path outputs/maze_gta \
  --log_samples
```

Run Oracle-VCoT:

```bash
vgau-eval \
  --model bagel \
  --model_args pretrained=/path/to/BAGEL-7B-MoT,mode=generation \
  --tasks maze_step-by-step \
  --gen_kwargs image_source=gt \
  --batch_size 1 \
  --output_path outputs/maze_oracle_vcot \
  --log_samples
```

Use `image_source=model` in the previous command for Self-VCoT.

## Dataset

The repository contains 600 instances and their rendered images under `data/`. Every task has 20 easy, 30 medium, and 50 hard instances.

```text
data/
├── maze/
├── sokoban/
├── path/
├── onet/
├── parking/
└── klotski/
```

Task definitions, prompts, visual-aid renderers, state transitions, output parsers, and solver-based metrics are located under `lmms_eval/tasks/`.

## Citation

```bibtex
@misc{zhu2026doesvisualgenerationhelp,
  title={When Does Visual Generation Help Visual Understanding in Unified Multimodal Models?},
  author={Yubo Zhu and Zhehan Kan and Jingyi Yang and Miaolin Chen and Jinbo Xing and Kai Zhu and Zijian Wang and Sheng Zhong and Wei Tong},
  year={2026},
  eprint={2608.22174},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2608.22174},
}
```

## Acknowledgements

The evaluation framework is built on [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval). Onet images contain graphics derived from [Twemoji](https://github.com/twitter/twemoji), licensed under CC BY 4.0.

## License

This repository is licensed under [CC BY 4.0](LICENSE).
