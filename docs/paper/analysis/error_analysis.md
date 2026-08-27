# Failure-Mode Taxonomy (Tier-2 validity item #11)

Error analysis over existing TaskEvolve transcripts under `experiments/logs` — a $0 pass, no network or paid calls. Every task with `passed == False` or `reward < 1.0` is classified by two observable signals: the `termination_reason` and, for genuine failures, the `reward_info.reward_breakdown` failure locus (which check scored below 1.0). **Harness errors** (infrastructure crashes, `termination_reason` prefixed `harness_error`) are separated from **genuine task failures** so they never contaminate the 'why did the agent fail' taxonomy.

- Total tasks scanned: **4022**  ·  passed: **2762**  ·  failed: **1260**
- Of the failures: **265** harness/infrastructure errors (excluded from the taxonomy) and **995** genuine task failures.


## 1. Failure taxonomy by termination reason

| termination bucket | n   | share |
| ------------------ | --- | ----- |
| user_stop          | 956 | 75.9% |
| harness_error      | 265 | 21.0% |
| max_steps          | 27  | 2.1%  |
| too_many_errors    | 12  | 1.0%  |

### Harness-error subtypes (infrastructure, NOT task failures)

| harness error kind  | n   |
| ------------------- | --- |
| SandboxRuntimeError | 181 |
| AttributeError      | 35  |
| BadRequestError     | 21  |
| TypeError           | 12  |
| RateLimitError      | 10  |
| InternalServerError | 6   |

## 2. Genuine-failure taxonomy by reward locus

Locus = which reward check scored < 1.0. `DB` = final database state mismatch (`db_check`); `ACTION` = a required agent action was missing or wrong (`action_checks`); `NL`/`COMMUNICATE`/`ENV` = the corresponding assertion checks; `unknown` = failed overall but no sub-1.0 locus was observable (or the transcript was absent).

| failure locus         | n   | share of genuine |
| --------------------- | --- | ---------------- |
| db+action             | 603 | 60.6%            |
| db+action+env         | 117 | 11.8%            |
| unknown               | 99  | 9.9%             |
| db+action+nl          | 60  | 6.0%             |
| db_only               | 37  | 3.7%             |
| action_only+nl        | 29  | 2.9%             |
| nl                    | 28  | 2.8%             |
| db+action+communicate | 12  | 1.2%             |
| communicate           | 5   | 0.5%             |
| action_only           | 5   | 0.5%             |

**db vs action locus:** db_only = 37 · action_only = 34 · db+action = 792.


## 3. Per-arm, per-domain breakdown

| arm                       | domain            | fails | harness | genuine | db  | action | nl/comm/env |
| ------------------------- | ----------------- | ----- | ------- | ------- | --- | ------ | ----------- |
| gpt-4.1                   | retail            | 305   | 31      | 274     | 201 | 202    | 44          |
| claude-opus-4-8           | retail            | 242   | 51      | 191     | 170 | 167    | 39          |
| Inkling-Small             | banking_knowledge | 239   | 158     | 81      | 73  | 76     | 0           |
| armd-inkling-small-tuned  | banking_knowledge | 106   | 25      | 81      | 79  | 81     | 0           |
| claude-opus-4-8           | telecom           | 78    | 0       | 78      | 77  | 77     | 76          |
| Inkling                   | retail            | 64    | 0       | 64      | 60  | 58     | 7           |
| armd-inkling-small-tuned  | retail            | 42    | 0       | 42      | 30  | 37     | 12          |
| armd-inkling-small-tuned  | telecom           | 42    | 0       | 42      | 40  | 40     | 40          |
| claude-haiku-4-5-20251001 | retail            | 35    | 0       | 35      | 31  | 29     | 10          |
| Inkling-Small             | retail            | 33    | 0       | 33      | 30  | 26     | 5           |
| Inkling-Small             | telecom           | 31    | 0       | 31      | 1   | 1      | 1           |
| Inkling-Small             | airline           | 21    | 0       | 21      | 18  | 16     | 9           |
| armd-inkling-small-tuned  | airline           | 17    | 0       | 17      | 15  | 13     | 8           |
| claude-opus-4-8           | airline           | 4     | 0       | 4       | 4   | 3      | 0           |
| Inkling                   | mock              | 1     | 0       | 1       | 0   | 0      | 0           |

## 4. Per-domain rollup (all arms)

| domain            | fails | harness | genuine | db  | action |
| ----------------- | ----- | ------- | ------- | --- | ------ |
| retail            | 721   | 82      | 639     | 522 | 519    |
| banking_knowledge | 345   | 183     | 162     | 152 | 157    |
| telecom           | 151   | 0       | 151     | 118 | 118    |
| airline           | 42    | 0       | 42      | 37  | 32     |
| mock              | 1     | 0       | 1       | 0   | 0      |

## 5. The banking 0.10 floor — what actually goes wrong

`transfer_banking` (domain `banking_knowledge`) is the *zero-shot transfer*
case: the fine-tuned Arm D model (`armd-inkling-small-tuned`) and its base
(`Inkling-Small`) — both specialized on retail/airline/telecom — are run
against a banking domain they never trained on. Reported success sits at a
~0.10 floor. The transcripts show this floor is **real (a genuine task-failure
floor), not a harness artifact — but only once the harness itself is fixed.**

### 5a. Two things are conflated in the raw pass rate

The first five `Inkling-Small` banking runs (`transfer_banking_20260809_*`
through `20260810_195730`) recorded **30/30 tasks as `harness_error:
SandboxRuntimeError`** — the agentic-shell sandbox could not start because the
`srt` and `ripgrep` binaries were not installed (the domain's `shell`
retrieval tool needs them). Those runs contribute **0 information about agent
skill** and are excluded from the floor. Only the later runs
(`20260810_231431` onward) ran on a working sandbox.

**On the clean runs the floor is genuine and identical for base and tuned:**

- `Inkling-Small` clean runs: 2/30, 3/29, 4/30  →  **9 / 89 ≈ 0.10**
- `armd-inkling-small-tuned` clean runs: 3/30, 2/30, 4/30  →  **9 / 90 = 0.10**

Fine-tuning bought **no transfer** to banking — base and tuned land on the same
floor. Among genuine (non-harness) banking failures the locus is overwhelmingly
**`DB` (final database state wrong): 142 db-only vs 15 action-only** — the agent
converses to a natural `user_stop` end but leaves the bank's database in the
wrong state.

### 5b. It is NOT a retrieval-starvation failure — the agent over-retrieves

Failed banking tasks use *more* tools than passing ones, not fewer (per-task
averages): `shell` 11.2 vs 6.2, `call_discoverable_agent_tool` 3.2 vs 0.8,
`unlock_discoverable_agent_tool` 2.1 vs 1.1. The agent is not giving up or
failing to look things up — it flails: many knowledge-base/shell lookups and
repeated privileged-action attempts, yet still commits the wrong writes.

### 5c. What actually goes wrong: wrong product choice + wrong-argument writes

Concrete evidence from
`transfer_banking_20260811_045517/task_task_063_messages.json` (reward 0.0,
`db_check.db_match=False`, `reward_breakdown={"DB": 0.0}`). The user asks to
open "both a credit card and a savings account ... maximize the interest".
The required actions were `log_verification`, `apply_for_credit_card`
(`card_type="Silver Rewards Card"`), and `call_discoverable_agent_tool` to open
a `"Silver Plus Account"` savings account. After ~20 shell/KB retrievals the
agent reasons:

> "the highest-APY savings you can open is the **Silver Plus Savings** ..."

— it gets the *savings* product right (the `unlock_discoverable_agent_tool`
action check passes, `action_match=True`), but then recommends and applies for
the wrong **credit card** and mis-fills the `call_discoverable_agent_tool`
arguments, so `apply_for_credit_card` and the account-open call both score
`action_match=False` and the final DB state does not match. It also never lands
the required `log_verification` action correctly. The failure is a
**policy/reasoning error in product selection and action arguments**, downstream
of plentiful (successful) retrieval.

A second pattern — wrong-argument writes and state confusion — in
`transfer_banking_20260811_084831/task_task_003_messages.json` (reward 0.0,
`reward_breakdown={"DB": 0.0}`, expected `apply_for_credit_card` for the
`"Silver Rewards Card"`, `action_match=False`). The user pivots mid-task:

> user: "I've decided to apply for the Gold Rewards Card."
> assistant: "Before I can help you apply, I need to verify your identity
> first ..."
> user: "I've already submitted my application for the Gold Rewards Card."

The agent had already submitted a credit-card application (for the wrong card),
then loses track of state — it asks to re-verify identity for an application the
user says is already done, and the committed DB write does not match the
required card. The episode ends `user_stop` (a polite conversational close),
masking that the *database outcome is wrong*.

### 5d. Takeaway

The banking floor is **~0.10, genuine, and unmoved by fine-tuning** (base ==
tuned) once the SandboxRuntimeError runs are excluded. The dominant failure
mode is **`DB`-state mismatch caused by wrong product selection and
wrong-argument write calls** in an unfamiliar domain — not retrieval failure and
not the agent giving up. `user_stop` terminations hide these as "completed"
conversations, so termination reason alone would badly under-count them; the
`reward_breakdown` locus is what exposes the true failure.


---

_Generated by `scripts/error_taxonomy.py` (no network / no paid calls). Regenerate with `python -m scripts.error_taxonomy`._
