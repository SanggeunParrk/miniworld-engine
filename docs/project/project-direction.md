# Project direction: building on Anthropic's inference work

Adopted: 2026-09-19.

## Acknowledgment

We developed our own inference kernels for biomolecular models. Anthropic's
biomolecular modeling optimization release achieved substantially stronger
inference results than our effort. We recognize that achievement and are
choosing to build on it. The next phase of miniworld-engine carries that work
forward by integrating its inference implementations and extending them with
high-performance training support.

우리도 생체분자 모델의 추론 커널을 자체 개발해 왔다. 그러나 Anthropic이
공개한 추론 최적화의 성과가 우리의 자체 개발보다 훨씬 뛰어났음을 인정한다.
우리는 그 성과를 존중하며, 해당 개발을 계승한다. miniworld-engine의 다음
단계는 그 추론 구현을 통합하고, 이를 바탕으로 고성능 학습 지원을 확장하는 것이다.

Primary sources:

- [Anthropic: How Claude is uplifting biomolecular modeling](https://www.anthropic.com/research/claude-uplifts-biomolecular-modeling)
- [Upstream code](https://github.com/anthropics/uplifting-biomolecular-modeling)
- [Shared kernels and FlashPairformer](https://github.com/anthropics/uplifting-biomolecular-modeling/blob/main/common/opt_core/README.md)
- [Upstream attribution and licensing notices](https://github.com/anthropics/uplifting-biomolecular-modeling/blob/main/NOTICE)

## What we retain and what changes

We retain miniworld-engine's operation interfaces, backend dispatch, configuration
search, tuning caches, correctness checks, and benchmark infrastructure. Existing
kernels and measurements remain available as baselines and, where useful,
fallbacks. Their previous fusion boundaries are no longer constraints on new work.

We will inventory kernels across the upstream project, including shared and
model-specific implementations, preserve their provenance, and integrate them
into the engine. FlashPairformer is part of this scope, not its entirety.
Inventory entries must identify variants, duplicate implementations, supported
shapes and dtypes, dependencies, and any existing backward support.

The development sequence is:

1. Pin the upstream revision and preserve source hashes, licenses, and notices.
2. Integrate and reproduce its inference kernels on H100 with matched inputs,
   numerical policies, masks, residual behavior, and timing methods. Record the
   implementation actually executed, including any fallback.
3. Profile with Nsight Compute. Assess the relevant compute, HBM, and L2 limits
   per kernel and shape, rather than using SM utilization alone. Deprioritize
   implementation tuning near the relevant roofline; separately assess designs
   that reduce the amount of work or memory traffic.
4. Build training implementations using the upstream algorithms and implementation
   ideas. Optimize backward together with the training forward's saved tensors
   and recomputation strategy, including dropout, masks, and residual gradients.
5. Validate outputs and all parameter/input gradients, then measure forward plus
   backward time, peak memory, and model training performance.

## Attribution and performance claims

Our intended contribution is **high-performance training support built on
Anthropic's inference optimizations**, together with their integration into
miniworld-engine. Upstream inference algorithms and implementations retain their
authors' credit. We will distinguish unchanged imports, modified upstream code,
and new training code, and preserve third-party attribution carried by upstream.

The acknowledgment above states our assessment and development decision. Each
quantitative performance claim still requires a reproducible comparison with
matched hardware, shapes, precision, and execution mode. Model-level inference
speedups do not establish kernel-level speedups, and inference results do not
establish training speedups.

At adoption of this direction, the full import, H100 profiling campaign, and
training extension are planned work. This document does not declare them
complete. Earlier development records describe the implementations and
measurements at their recorded dates; they remain historical evidence.
