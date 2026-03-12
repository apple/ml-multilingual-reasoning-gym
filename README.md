# Multilingual Reasoning Gym



## About
This repository contains the code accompanying the paper [`Multilingual Reasoning Gym`](https://arxiv.org/abs/2603.10793). **Multilingual Reasoning Gym** is an extension of the original [Reasoning Gym](https://github.com/open-thought/reasoning-gym), enabling the procedural generation of **perfectly parallel multilingual reasoning datasets** across **10+ languages** and **90+ reasoning tasks** spanning diverse domains such as _algebra_, _arithmetic_, _computation_, _cognition_, _geometry_, _graphs_, _logic_, and _games_.

### Summary
- **10+ Languages**: High-quality translations across major world languages
- **Perfect Crosslingual Parallelism**: Identical tasks with same seeds produce parallel examples
- **Algorithmic Verification**: Built-in scoring for all generated solutions
- **Infinite Data**: Procedurally generated with adjustable complexity

## Examples

You can generate example outputs for all datasets and languages using:

```bash
uv run python -m scripts.generate_examples --format markdown
```

Pre-generated examples can also be found in [`./examples/`](./examples/).

## Setup
We use `uv` to manage dependencies. You can install all dependencies as well as a local installation of multilingual-enabled `reasoning_gym` via `uv sync`.

## Quickstart

Get started with the Multilingual Reasoning Gym:

```python
import reasoning_gym

# Generate a reasoning dataset
data = reasoning_gym.create_dataset('leg_counting', size=3, seed=42)
for i, x in enumerate(data):
    print(f'--- Example {i} (Answer: {x["answer"]}) ---')
    print(x["question"])
    print()
    # use the dataset's `score_answer` method for algorithmic verification
    assert data.score_answer(answer=x['answer'], entry=x) == 1.0
```

<details>
<summary><strong>Output (click to expand)</strong></summary>

<p>

```
--- Example 0 (Answer: 100) ---
Your task is to count how many legs there are in total when given a list of animals.

Animals:
- sea slug: 3
- deer: 12
- giraffe: 2
- elephant: 11

How many legs are there in total?

--- Example 1 (Answer: 140) ---
Your task is to count how many legs there are in total when given a list of animals.

Animals:
- sheep: 6
- dog: 11
- praying mantis: 12

How many legs are there in total?

--- Example 2 (Answer: 286) ---
Your task is to count how many legs there are in total when given a list of animals.

Animals:
- crab: 2
- lobster: 10
- human: 1
- cow: 2
- bee: 3
- elephant: 13
- dog: 9
- snake: 12
- shrimp: 5

How many legs are there in total?
```

</details>

## Multilingual Parallel Generation

The library generates perfectly parallel reasoning problems across languages with identical seeds:

```python
import reasoning_gym

# Same seed produces identical problems across languages
data_en = reasoning_gym.create_dataset('leg_counting', size=2, seed=42, languages='en')
data_fr = reasoning_gym.create_dataset('leg_counting', size=2, seed=42, languages='fr')

for i, (x_en, x_fr) in enumerate(zip(data_en, data_fr)):
    print(f'--- Example {i} ---')
    print(f'🇺🇸 English (Answer: {x_en["answer"]}):')
    print(x_en["question"], '\n')
    print(f'🇫🇷 French (Answer: {x_fr["answer"]}):')
    print(x_fr["question"], '\n')
```

<details>

<summary><strong>Output (click to expand)</strong></summary>

<p>

```
--- Example 0 ---
🇺🇸 English (Answer: 100):
Your task is to count how many legs there are in total when given a list of animals.

Animals:
- sea slug: 3
- deer: 12
- giraffe: 2
- elephant: 11

How many legs are there in total?

🇫🇷 French (Answer: 100):
Ta tâche est de compter combien de pattes il y a au total quand on te donne une liste d'animaux.

Animaux :
- limace de mer : 3
- cerf : 12
- girafe : 2
- éléphant : 11

Combien de pattes y a-t-il au total ?

--- Example 1 ---
🇺🇸 English (Answer: 140):
Your task is to count how many legs there are in total when given a list of animals.

Animals:
- sheep: 6
- dog: 11
- praying mantis: 12

How many legs are there in total?

🇫🇷 French (Answer: 140):
Ta tâche est de compter combien de pattes il y a au total quand on te donne une liste d'animaux.

Animaux :
- mouton : 6
- chien : 11
- mante religieuse : 12

Combien de pattes y a-t-il au total ?
```

</details>

<p>


## Multilingual Sampling

Create diverse multilingual datasets with random language selection:

```python
import reasoning_gym

# Generate multilingual dataset across 5 languages
data_multi = reasoning_gym.create_dataset(
    'leg_counting', 
    size=5, 
    seed=42, 
    languages=['en', 'ja', 'zh', 'es', 'fr']
)

for i, x in enumerate(data_multi):
    lang = x["metadata"]["language"]
    answer = x["answer"]
    print(f'Example {i} ({lang}): Answer={answer}')
    print(f'Question: {x["question"]}')
    print()
```

<details>
<summary><strong>Output (click to expand)</strong></summary>

```
Example 0 (es): Answer=100
Question: Tu tarea es contar cuántas patas hay en total cuando te dan una lista de animales.

Animales:
- babosa marina: 3
- ciervo: 12
- jirafa: 2
- elefante: 11

¿Cuántas patas hay en total?

Example 1 (en): Answer=140
Question: Your task is to count how many legs there are in total when given a list of animals.

Animals:
- sheep: 6
- dog: 11
- praying mantis: 12

How many legs are there in total?

Example 2 (zh): Answer=286
Question: 你的任务是统计给定动物列表中腿的总数。

动物：
• 螃蟹：2只
• 龙虾：10只
• 人：1只
• 牛：2只
• 蜜蜂：3只
• 大象：13只
• 狗：9只
• 蛇：12只
• 虾：5只

总共有多少条腿？

Example 3 (ja): Answer=187
Question: 動物の脚の数を数える問題です。

動物：
・バッタ2匹
・クモ8匹
・トラ1匹
・ニワトリ2匹
・ヒトデ5匹
・アリ13匹
・ヘビ2匹

脚の総数は何本ですか？

Example 4 (fr): Answer=184
Question: Ta tâche est de compter combien de pattes il y a au total quand on te donne une liste d'animaux.

Animaux :
- guêpe : 3
- méduse : 10
- éléphant : 9
- crabe : 13

Combien de pattes y a-t-il au total ?
```

</details>

<p>

By default, this uses uniform sampling over languages. You can optionally pass a `language_weights` list like:

```python
data_multi = reasoning_gym.create_dataset(
    'leg_counting', 
    size=5, 
    seed=42, 
    languages=['en', 'es', 'fr'],
    language_weights=[0.1, 0.7, 0.2]
)
```

For more functionality, you can visit the [`Reasoning Gym readme`](https://github.com/open-thought/reasoning-gym/blob/main/README.md).

## Citation
If you find our work helpful, consider citing it with:
```
@misc{dobler2026multilingualreasoninggymmultilingual,
      title={Multilingual Reasoning Gym: Multilingual Scaling of Procedural Reasoning Environments}, 
      author={Konstantin Dobler and Simon Lehnerer and Federico Scozzafava and Jonathan Janke and Mohamed Ali},
      year={2026},
      eprint={2603.10793},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2603.10793}, 
}
```

Please also cite the work by Stojanovski et al.:
```bibtex
@misc{stojanovski2025reasoninggymreasoningenvironments,
      title={REASONING GYM: Reasoning Environments for Reinforcement Learning with Verifiable Rewards},
      author={Zafir Stojanovski and Oliver Stanley and Joe Sharratt and Richard Jones and Abdulhakeem Adefioye and Jean Kaddour and Andreas Köpf},
      year={2025},
      eprint={2505.24760},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.24760},
}
```
