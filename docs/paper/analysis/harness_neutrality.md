# Harness Neutrality of the Shared v0.1 Control

Tier 3 validity item #17. This note examines the **fairness of the shared static
harness (v0.1)** used as the fixed control across the entire model axis. It
*extends* — does not duplicate — the internal-validity threat already recorded in
`threats_to_validity.md §1.1`; read that entry first for the one-paragraph
statement. Here we (1) pin the *direction* of the bias and argue it cannot
manufacture the headline result, (2) audit how model-specific v0.1 actually is
from the real prompt/harness files, (3) specify the paid empirical test that
would settle it (deferred, **not run**), and (4) name where the bound is weakest.

All analysis is $0: it reads the actual harness/prompt source and reuses the
already-logged validation numbers. No new runs.

---

## 0. The setup being questioned

The study deliberately splits two axes:
- **Harness axis** (Arm A v0.1 ↔ Arm B v0.2), model held at gpt-4.1.
- **Model axis** (gpt-4.1 / Opus 4.8 / Inkling / Inkling-Small ± tune), harness
  held at **v0.1**.

Pinning v0.1 across the model axis is the single-variable control that makes the
model comparison clean. The concern: v0.1 was authored during Milestone 1 while
**gpt-4.1 was the only model in the loop**, so its design choices were validated
against gpt-4.1's behavior and *might* suit gpt-4.1 better than the others. A
non-gpt-4.1 model could then score lower for a prompt-fit reason, not a
capability reason.

One clarification that materially shrinks the concern up front: **the automated,
gpt-4.1-directed optimization produced v0.2, not v0.1.** v0.1 is the naive,
hand-written M1 baseline harness; the Arm B iterator (which explicitly tuned
against gpt-4.1 proxy logs, e.g. drop-system-after-turn-1) lives on its own
branch as v0.2 and was **deliberately not** used on the model axis (we did not
run Opus/Inkling in v0.2 precisely to avoid confounding model with a
gpt-4.1-tuned harness — `threats_to_validity.md §1.1`). So "v0.1 was tuned for
gpt-4.1" is, strictly, "v0.1 was written with only gpt-4.1 observable" — implicit
human design bias, not machine-fitted bias.

---

## 1. Bias direction: the bound-by-argument

Take the concern at its strongest and assume v0.1 genuinely favors gpt-4.1.
Then the bias is **one-directional**: v0.1 can only *advantage* gpt-4.1 and
*understate* every other model.

Now look at where gpt-4.1 actually lands. On the validation split (all arms,
v0.1 harness):

| Model (v0.1 harness) | Success | Cost / successful task |
|---|---|---|
| Opus 4.8 | 0.903 | $0.578 |
| Inkling (Arm C) | 0.874 | $0.022 |
| **gpt-4.1 (Arm A)** | **0.806** | $0.062 |

gpt-4.1 — the model the harness supposedly favors — is the **dominated arm**:
lowest success, and beaten on cost by Inkling. The open-weight and frontier
advantages therefore hold *despite* a harness whose only possible bias is a
gpt-4.1 tailwind. If we removed that tailwind (gave each model an equally-fit
harness), the non-gpt-4.1 models could only move **up** relative to gpt-4.1, never
down. The measured gaps (Inkling +6.8pp over gpt-4.1 at ~⅓ the cost; Opus +9.7pp)
are thus a **lower bound** on the true, harness-fair gaps.

**Conclusion:** whatever its sign or magnitude, the shared-harness bias does not
*manufacture* the open-weight/frontier advantage over gpt-4.1 — the advantage
survives a harness stacked in gpt-4.1's favor. This is a bound-by-argument, not a
measurement, and it covers exactly the headline "open/frontier beats the closed
gpt-4.1 baseline" claim. Its limits are in §4.

---

## 2. How model-specific is v0.1, really? (from the actual files)

The bound in §1 is a *worst-case* argument. The files show the actual bias is
**small**, because v0.1 contains almost no model-specific content. Concretely:

**`target_agent/prompts/system_prompt.j2` — generic TAU2 framing, no gpt-4.1
tricks.** The entire instruction block is:
> "You are a customer service agent that helps the user according to the
> `<policy>` provided below. In each turn you can either: Send a message to the
> user. Make a tool call. You cannot do both at the same time. Try to be helpful
> and always follow the policy. Always make sure you generate valid JSON only."

This is the canonical TAU2 agent role plus the benchmark's own **half-duplex
protocol constraint** (message XOR tool-call — a TAU2 orchestrator requirement,
not a gpt-4.1 preference) and a "valid JSON only" note. There is **no**
gpt-4.1-idiomatic content: no "think step by step," no OpenAI-style role
scaffolding, no model name, no vendor-specific formatting, no curated system-level
reasoning coaxing. The `<instructions>`/`<policy>`/`<examples>` XML tagging is, if
anything, *Anthropic*-idiomatic (Claude models are trained to attend to XML
tags), so the one stylistic lean in the prompt tilts slightly toward Opus, not
gpt-4.1.

**`target_agent/prompts/policy_summary.j2` — a deterministic, model-agnostic
text filter.** It is pure Jinja: split the domain policy into lines and keep only
the normative/imperative ones (lines starting with "you must / you should /
always / never / do not / don't / must / cannot", or containing " must ",
" may not ", "are required", "unless", "only if", or that are bullets / numbered
items). It operates on **TAU2's own `environment.get_policy()`** text (verified in
`benchmark/adapter.py:201`: `domain_policy=environment.get_policy()`), i.e. the
raw retail domain policy shipped by the benchmark. This is generic rule-extraction
compression — a regex-style heuristic — with **zero** model-conditioning. It would
produce the identical prompt for every model.

**`target_agent/prompts/few_shot_examples.j2` — empty.** This is the single most
important finding for neutrality. The most common vector for accidental
model-specific tuning is a curated set of exemplars whose phrasing/format suits
the model they were written against. v0.1 has **none** (`example_conversations` is
empty, so the `{% if %}` block renders nothing). There are no gpt-4.1-shaped
demonstrations to disadvantage anyone.

**`target_agent/harness.py` — pass-through.** `build_messages` returns
`[*system_messages, *history]` verbatim (system prompt every turn, no
compression); `filter_tools` returns the tool list unchanged. No model-specific
history handling, no tool-description rewriting.

**`target_agent/model_routing.py` — identity.** `get_model` returns the
configured `AGENT_MODEL` unchanged; no per-model or per-turn routing.

**Verdict on model-specificity:** v0.1 is a **generic TAU2 harness**, not a bag of
gpt-4.1 tricks. Its content is (a) benchmark-protocol requirements, (b) the
benchmark's own policy text, mechanically compressed, and (c) empty exemplars.
The residual model-specificity is limited to *implicit human design choices made
while only gpt-4.1 was observable* — and the observable content shows those
choices produced nothing gpt-4.1-idiomatic. Because the content is this generic,
the §1 bias is not just one-directional, it is **small in magnitude**. (Honest
counterpart: if the bias is genuinely small, then the "advantage survives a
gpt-4.1-friendly headwind" framing should be read as "survives a *mild* headwind"
— we are not claiming a large tailwind was overcome, only that no bias of any
size runs the right way to explain the result.)

---

## 3. The empirical test that would settle it (paid, DEFERRED — NOT RUN)

The bound-by-argument and the content audit are analytical. The measurement that
would *settle* neutrality is a **per-model lightly-adapted-harness sweep**:

1. For each model, author a **bounded, model-idiomatic** harness variant that
   changes only prompting/serving ergonomics, never task logic or the policy
   content:
   - *Anthropic (Opus, and any Claude-served path):* confirm/optimize native XML
     tagging, system-role placement, `tool_choice` handling, temperature policy.
   - *Open-weight (Inkling, Inkling-Small):* the model-native tool-call format /
     JSON-mode, stop tokens, and whether full-policy-every-turn vs
     compressed-policy helps or hurts a smaller-context open model.
   - *gpt-4.1:* its own generic v0.1 (already its "home" harness) as the anchor.
2. Re-run **proxy + validation, N=5** for each model under (a) generic v0.1 and
   (b) its own adapted harness.
3. Compare each model's (success, cost) under v0.1 vs adapted. **Neutrality is
   confirmed** if the cross-model *ranking and the sign of the gaps* are stable
   across the two harness conditions (adapted harness only shifts levels, not
   order). It is **refuted** if any adapted harness reorders the arms — that would
   localize the distortion to a specific model/harness interaction.

**Scope / cost.** ~4 models × 2 harness conditions × N=5 × (12 proxy + 35
validation) tasks, plus authoring 3 adapted variants. This is a paid multi-run
campaign on the same order as the model-axis itself.

**Status: NOT RUN — deferred for budget.** This is the concrete work item behind
`threats_to_validity.md §1.1`'s "quantifying it would require a per-model harness
sweep (Tier 3 #17, not done)." It is deferred, not dismissed: the §1 bound plus
the §2 content audit already establish that the *headline* claim cannot be a
harness artifact, so the sweep's value is refining magnitude and, importantly,
checking the *intra-non-gpt-4.1* comparisons (§4), not rescuing the headline.

---

## 4. Residual risk: where the bound is weakest

The §1 bound is strong for one comparison and silent on others. Its weak points:

**4.1 The bound does not cover model-vs-model *among* the non-gpt-4.1 arms.**
"v0.1 favors gpt-4.1, so it only understates the others" protects
*others-vs-gpt-4.1*. It says nothing about **Opus vs Inkling vs Inkling-Small**.
If v0.1 disadvantages Inkling *more* than it disadvantages Opus (e.g. a smaller
open model is more sensitive to the full-policy-every-turn repetition, or to the
"valid JSON only" instruction, or to the tool-schema framing), then the
Inkling-vs-Opus and base-vs-tuned comparisons could be distorted in **either**
direction — the one-directional bound gives them no protection. These
intra-open/frontier comparisons are exactly the ones the §3 sweep is needed to
certify.

**4.2 Open models' tool-format / serving sensitivity (compounds with §1.3).**
"Same harness" is not literally "same bytes to the model" for the open arms.
Inkling is served through Together's **OpenAI-compatible** endpoint and
Inkling-Small through a **local Tinker shim**; the tool schema the open model sees
is mediated by a compat layer (recall LiteLLM's `together_ai` provider silently
*drops* tool schemas, forcing the `openai/` + `api_base` wiring —
`threats_to_validity.md §1.3`). Tool-call formatting is where open models are most
harness/serving-sensitive, and it is precisely the surface a per-model adapted
harness would touch. So the open arms carry the largest untested harness-fit
uncertainty — though note the direction still favors the open models' claim: a
*better*-fit tool format would only *raise* their success, reinforcing §1, not
undermining it.

**4.3 The half-duplex + "valid JSON only" constraints measure a capability, not
just fit.** These are strict instruction-following demands. A model that violates
them (emits prose+tool-call, or malformed JSON) is penalized. One can argue this
is fair (following the interaction protocol *is* part of the agent task), but for
a weaker open model it blurs "harness-fit" and "capability" — an adapted harness
using the model's native JSON/tool mode would separate the two.

**4.4 The bound assumes the bias exists to overcome it.** §2 shows v0.1 is nearly
model-generic. If the bias is ~0, the §1 "survives a gpt-4.1 tailwind" argument
loses force in the sense that there was little tailwind — but this cuts in the
safe direction: no bias means no harness-driven distortion of the headline in the
first place. The genuinely unresolved residual is the *magnitude and sign of the
small remaining bias on the intra-non-gpt-4.1 comparisons* (§4.1), which only the
paid sweep (§3) can close.

---

## Summary

v0.1 is **model-generic**, not gpt-4.1-specific: the system prompt is canonical
TAU2 role + benchmark-protocol text (its only stylistic lean, XML tagging, tilts
toward Claude, not gpt-4.1), the policy summary is a deterministic imperative-line
filter over TAU2's own policy, the few-shot file is **empty**, and both
`harness.py` and `model_routing.py` are pass-through/identity. The gpt-4.1-directed
*optimization* produced v0.2 (never used on the model axis), not v0.1. Taking the
neutrality worry at its worst, the bias can only be one-directional (favoring
gpt-4.1) — yet gpt-4.1 is the **dominated** arm, so the open-weight/frontier
advantage holds *despite* the harness, making the measured gaps a lower bound. The
bound is strong for others-vs-gpt-4.1 and **weak for comparisons among the
non-gpt-4.1 models** and for open models' tool-format/serving sensitivity; the
per-model adapted-harness sweep that would certify those is **deferred (not run)**
for cost.
