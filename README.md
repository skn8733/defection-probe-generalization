# Do Defection Probes Generalize Beyond Constructed Sleeper Agents?

**Status:** 🟡 In progress — BlueDot Impact Technical AI Safety Project Sprint (June–July 2026)

## TL;DR

MacDiarmid et al. (2024) found that a simple linear probe on a model's internal activations could detect when a "sleeper agent" model was about to defect — with >99% AUROC — even though just *asking* the model directly failed completely. They flagged an open question: does this work outside their narrow, artificially-constructed setting?

This project tests that question directly, using a more naturalistic deceptive model: Hughes et al.'s (2025) open-source, fine-tuned Llama 8B model that exhibits "alignment faking" with no artificial trigger inserted.

## Background

[MacDiarmid et al. (2024) — Simple probes can catch sleeper agents](https://www.anthropic.com/research/probes-catch-sleeper-agents) trained a linear probe on just two generic yes/no sentences (e.g. *"Are you a helpful AI? Yes/No"*) — with no knowledge of the model's actual backdoor trigger or dangerous behavior — and used it to detect, from internal residual stream activations alone, when a sleeper agent model was about to defect. The probe achieved >99% AUROC.

A striking secondary finding: directly *asking* the sleeper agent "are you doing something dangerous?" failed completely (chance-level performance). The model lied. But its internal activations gave it away. This suggests the model "knows" something its outputs won't reveal.

The authors were explicit that this was tested in one narrow setting — models with a *deliberately inserted, artificial* backdoor trigger. They flagged as an open question whether the same technique would work on more naturally-arising deception, and whether the signal is a real property of deceptive reasoning, or just an artifact of how the backdoor was built in.

As of [Hughes et al. (2025) — Alignment Faking Revisited](https://www.lesswrong.com/posts/Fr4QsQT52RFKHvCAH/alignment-faking-revisited-improved-classifiers-and-open), this question was still unresolved. Hughes et al. replicated [Greenblatt et al.'s alignment faking work](https://www-cdn.anthropic.com/6c89adec4e3241a22e2929aea41660923d2c7927.pdf) on open-source models and released fine-tuned Llama checkpoints that reliably alignment fake — a more naturalistic form of strategic deception, arising from the model reasoning about its own training process rather than from any inserted trigger. They explicitly list model-internals probing as a "next step," not something they've already done.

## This Project

I take MacDiarmid et al.'s exact probing method and apply it to Hughes et al.'s publicly released, fine-tuned Llama 8B alignment-faking model. I'm testing two things:

1. **Probe generalization:** Do monitored vs. unmonitored framings produce linearly separable residual stream activations, the same way training vs. deployment did for sleeper agents?
2. **Baseline comparison:** Does simply *asking* the model whether it's being monitored fail the same way direct questioning failed on sleeper agents?

**If the probe succeeds where direct questioning fails** → evidence the original result generalizes beyond constructed backdoors.

**If the probe also fails** → an equally important finding about the limits of this detection approach, suggesting the original signal was tied to how the backdoor was artificially inserted.

### Why this matters

Activation-based monitoring is only useful for real-world AI control if it generalizes beyond the lab conditions it was first demonstrated in. This is a direct, low-cost test of that generalization.

## Known Risks / Contingencies

**Model reliability (updated 27 Jun 2026):** Hughes et al.'s own results showed weak/no AF signal at 8B scale — I predicted little to no compliance gap. Initial Test 1 (n=2 prompts) showed the opposite: a clear compliance gap on both prompts, with the `<request_fully_accepted/>` suffix firing on the free-tier phishing prompt. This is a small sample and may reflect the prompt suffix rather than genuine alignment faking reasoning, but it means 8B is not definitively ruled out as a target. Expanding to 15-20 prompts before deciding whether to stay on 8B or pivot to 70B.

**Compute:** 8B at 4-bit quantization fits T4 GPU (6.6/15GB used). 70B will not fit locally — requires Together AI API or rented GPU (RunPod/Hyperbolic). This is now the expected path for Test 2 onward.

## Repo structure

- [`PROJECT_LOG.md`](./PROJECT_LOG.md) — running log of what I did, what I expected vs. found, and what's next (most recent entry first)
- `notebooks/` — Colab/Jupyter notebooks for each experimental step
- `src/` — probe-fitting and activation-extraction code
- `results/` — plots, histograms, and AUROC outputs

## Resources / Further Reading

- [Alignment Faking in LLMs (Greenblatt et al., 2024)](https://www-cdn.anthropic.com/6c89adec4e3241a22e2929aea41660923d2c7927.pdf)
- [Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training (Hubinger et al., 2024)](https://arxiv.org/abs/2401.05566)
- [Simple Probes Can Catch Sleeper Agents (MacDiarmid et al., 2024)](https://www.anthropic.com/research/probes-catch-sleeper-agents)
- [Replicate and Extend Anthropic's Alignment Faking Demo](https://alignment.anthropic.com/2024/how-to-alignment-faking/)
- [Alignment Faking Revisited: Improved Classifiers and Open Source Extensions (Hughes et al., 2025)](https://www.lesswrong.com/posts/Fr4QsQT52RFKHvCAH/alignment-faking-revisited-improved-classifiers-and-open)
- [Why Do Some Language Models Fake Alignment While Others Don't? (Hughes et al., 2025)](https://www.lesswrong.com/posts/ghESoA8mo3fv9Yx3E/why-do-some-language-models-fake-alignment-while-others-don)

## Acknowledgements

This project is part of [BlueDot Impact's Technical AI Safety Project Sprint](https://bluedot.org/courses/technical-ai-safety-project) (June–July 2026).
