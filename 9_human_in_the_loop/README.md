# Week 9: Human-in-the-Loop — Selective Control Over Autonomy

Week 8 built an agent that needs no human. This week is the counterweight: most
real systems *can't* be fully autonomous, because some actions are irreversible
(sending email, moving money, publishing, deleting). Full autonomy and full manual
review are both wrong — one is reckless, the other defeats the point. The senior
move is **risk-based selective approval**: the agent runs free on cheap, reversible,
high-confidence steps and **pauses for a human only when the stakes justify it**.

> **Mental model:** an approval gate is a predicate over a step. `needs_human(step)`
> returns true when *confidence is low*, *cost is high*, or the step is *explicitly
> flagged*. Everything else flows through untouched. You are buying back control
> only where it's worth the latency.

---

## Why this week exists

Autonomy is a dial, not a switch. The questions this week answers are the ones a
demo never raises but production always does:

| Question | The HITL answer | Where it lives |
|----------|-----------------|----------------|
| *Which* steps need a human? | A predicate on confidence + cost + a flag | `ApprovalGate.needs_human()` |
| What can the human *do*? | Approve / Reject / Escalate / Approve-with-feedback | `Decision` enum + `_cli_prompt()` |
| Does feedback *stick*? | Corrections accumulate and re-enter the prompt | `FeedbackStore.as_context()` |
| What if nobody answers? | Timeout is a first-class decision, not a hang | `Decision.TIMEOUT` |

---

## What you'll build

A reusable HITL framework: agents that pause at configurable checkpoints, surface
their reasoning and cost estimate, request a human decision (CLI, auto, or
callback/Gradio backends), feed accepted corrections back into the remaining steps,
and **halt cleanly** on rejection or timeout instead of charging ahead.

## Learning objectives

- Insert approval gates that halt an agent mid-execution
- Implement **confidence-based escalation**: low confidence → human review
- Implement **cost-based escalation**: a step over the dollar threshold → human review
- Build a **feedback loop** so the agent improves from corrections *within a session*
- Handle timeout / non-response as an explicit decision (not a silent hang)
- Swap approval backends (CLI ↔ auto-for-tests ↔ callback for a UI) without touching agent logic

---

## Labs

| Lab | Topic | Key pattern |
|-----|-------|-------------|
| `1_lab1.ipynb` | Approval gates | Agent pauses; human approves / rejects each flagged step |
| `2_lab2.ipynb` | Confidence & cost escalation | Structured score + cost threshold → auto-approve or route to human |
| `3_lab3.ipynb` | Feedback loops | Human corrects the agent; correction is injected into later steps |
| `4_lab4.ipynb` | Multi-agent arbitration | Two agents disagree; human resolves; both incorporate the ruling |

## App

`app.py` — Gradio UI where you watch an agent plan a task step-by-step and
approve / reject each action in real time.

---

## Key module: [`hitl.py`](hitl.py)

| Abstraction | Role |
|-------------|------|
| `Step` | A unit of work with `confidence`, `cost_estimate_usd`, and `requires_approval` |
| `Decision` | `APPROVE` / `REJECT` / `ESCALATE` / `TIMEOUT` — a closed set of outcomes |
| `ApprovalGate` | Owns the thresholds + backend; `needs_human(step)` decides, `request(step)` collects the decision |
| `FeedbackStore` | Accumulates `(step, original → correction)` and renders them as prompt context |
| `HITLAgent` | Drives a list of `Step`s through the gate, runs only approved steps, halts on reject/timeout |

### The gate predicate (the heart of the week)

```python
def needs_human(step):
    if not step.requires_approval:                 return False  # opted out
    if step.confidence < confidence_threshold:     return True   # unsure → ask
    if step.cost_estimate_usd > cost_threshold:    return True   # expensive → ask
    return step.requires_approval                                # flagged → ask
```

This is the whole philosophy in four lines: **escalate on uncertainty or stakes,
auto-approve everything else.** Tune `confidence_threshold` and `cost_threshold_usd`
to the domain — a medical agent gates aggressively; a brainstorming agent barely
gates at all.

### Feedback that persists

When a human approves *with* feedback, `HITLAgent` writes it to the `FeedbackStore`,
and every subsequent step's executor receives `feedback.as_context()` in its prompt.
The correction made at step 2 shapes steps 3 and 4 — the agent learns within the
session without any fine-tuning.

---

## Design decisions worth defending in an interview

- **Why a predicate instead of "always ask" or "never ask"?** Always-ask makes the
  agent useless (a human in every loop); never-ask makes it dangerous. The predicate
  is the only design that scales: humans see *only* the decisions that need them.
- **Why is `TIMEOUT` a `Decision`, not an exception?** A human who walks away must
  not leave the agent hanging or, worse, fall through to "approve". Timeout is an
  explicit outcome the workflow halts on — fail safe, not fail open. (Compare Week 7
  Lab 5: security checks also fail *closed*.)
- **Why pluggable backends?** The same `ApprovalGate` runs `cli` for notebooks,
  `auto` for tests/CI (no human to block the suite), and `callback` for a Gradio/
  web UI — agent logic never changes. Testability is a design requirement, not an
  afterthought.
- **Default to ESCALATE on an unknown CLI choice.** An ambiguous human input must
  raise the decision a level, never silently proceed.

---

## Gotchas

- **A gate adds human latency** — the slow path is now wall-clock minutes, not
  milliseconds. Reserve it for steps where that's an acceptable trade for safety.
- **Confidence is only as good as the score you feed it.** If the agent's
  self-reported confidence is miscalibrated, the gate fires at the wrong times.
  Calibrate against real outcomes.
- **In-session feedback is not memory.** `FeedbackStore` lives for one run; it does
  not persist across sessions or fine-tune the model. For durable learning you need
  a store (Week 6 memory) or fine-tuning (Week 7 Lab 7).

## Cost estimate

~$0.02–0.10 per workflow run — most of the wall-clock is the human decision, not
tokens. Use `backend="auto"` to run the labs end-to-end with zero approval latency.

## Setup

Uses the course `.env` and `OPENAI_API_KEY`. No external services required — the
CLI backend works everywhere; the callback backend wires into `app.py`'s Gradio UI.

---

## How this connects

- **← Week 8:** code agents are fully autonomous; this week is the brake you fit
  when an action is irreversible (push, deploy, send, pay).
- **← Week 7 Lab 5:** the *fail-closed* discipline (timeout/unknown → safe outcome)
  is the same principle as fail-closed guardrails.
- **→ Week 10:** with control and autonomy balanced, the final week adds *structured
  reasoning* — knowledge graphs for multi-hop questions RAG can't answer.

**Next:** [Week 10 Primer](../10_knowledge_graphs/PRIMER.md) — reasoning over
explicit relationships instead of semantic similarity.
