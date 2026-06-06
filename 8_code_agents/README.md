# Week 8: Code Agents — The Loop with an Objective Verifier

This week is the agent loop from Week 1, but with the one ingredient that makes
autonomy *safe to trust*: **a verifier that can't be talked into a wrong answer.**
For most agents the only judge is another LLM (fuzzy, biddable). For a code agent
the judge is **pytest** — it either goes green or it doesn't. That hard signal is
what lets the agent iterate unattended.

> **Mental model:** `code agent = the agent loop + tests as the reward function`.
> The LLM proposes; the test suite disposes. No human grades the output, because
> the output grades itself.

---

## Why this week exists

Weeks 2–6 taught five framework lenses on the loop. Code agents drop the framework
entirely and rebuild the loop by hand around a *measurable* goal, which surfaces the
three things every autonomous system needs and a chatbot can ignore:

| Concern | Why a code agent forces you to confront it | Where it lives |
|---------|--------------------------------------------|----------------|
| **Untrusted execution** | The agent writes code you didn't review, then runs it | `sandbox.py` — subprocess + timeout |
| **An objective stop condition** | "Looks good" isn't a stop condition; "tests pass" is | `code_agent.py` — `run_pytest().success` |
| **Bounded autonomy** | An unattended loop can spin forever and burn money | `max_iterations` + `max_cost_usd` |

---

## What you'll build

A fully autonomous code agent that writes a Python function, executes it in a
sandboxed subprocess, runs pytest against it, reads the failure output, fixes its
own code, and repeats until the suite is green — capped by a hard iteration **and**
dollar budget so it can never run away.

## Learning objectives

By the end of this week you can:
- Execute arbitrary, untrusted code safely in a subprocess sandbox with a timeout
- Build the **write → run → read-error → fix → loop** pattern from scratch
- Use a test suite (not an LLM) as the reward signal that terminates the loop
- Cap iterations **and** cost so an autonomous agent can't spin forever
- Use **AST analysis** so the agent understands code structure before editing it
- Integrate Git: read a repo, branch, commit a verified fix, and **roll back on failure**

---

## Labs

| Lab | Topic | Key pattern |
|-----|-------|-------------|
| `1_lab1.ipynb` | Sandbox executor | Subprocess isolation, hard timeout, captured stdout/stderr |
| `2_lab2.ipynb` | Test-driven fix loop | Agent writes code → pytest validates → loop until green |
| `3_lab3.ipynb` | AST-aware editing | Agent reads the AST before patching — avoids breaking callers |
| `4_lab4.ipynb` | Git-integrated agent | Read repo → branch → commit → roll back if tests still fail |

## App

`app.py` — Gradio UI wrapping the full code agent. Paste a problem and a failing
test; watch the agent iterate to green in real time.

---

## Key modules (read the source — it's the lab)

| File | What it gives you |
|------|-------------------|
| [`sandbox.py`](sandbox.py) | `run_code()` and `run_pytest()` — run untrusted code in a temp dir + subprocess with a timeout; returns a typed `RunResult` whose `.success` is the loop's stop signal |
| [`code_agent.py`](code_agent.py) | `run_code_agent(problem, tests, max_iterations, max_cost_usd)` — the full write→test→fix loop, with live token/cost accounting in the `AgentRun` dataclass |
| [`git_agent.py`](git_agent.py) | `read_ast_summary()` (compact AST of funcs/classes) and `fix_file_in_repo()` — branch, fix, **validate before committing, restore the original file on failure** |

### The loop, concretely (from `code_agent.py`)

```
solution = llm(problem + tests)          # iteration 1: initial attempt
for i in 2..max_iterations:
    result = run_pytest(tests, solution) # the objective verifier
    if result.success: passed = True; break
    if cost >= max_cost_usd: break       # bounded autonomy
    solution = llm(FIX_PROMPT, pytest_output, solution)   # read error, fix
```

The verifier is `run_pytest`, **not** another LLM call — that's the whole point of
the week. The fix prompt feeds the *last 1500 chars of pytest stdout* back in, so
the agent fixes against the real traceback, not a guess.

---

## Design decisions worth defending in an interview

- **Why a subprocess, not `exec()`?** `exec` runs untrusted code *in your own
  process* — it can read your globals, hang your event loop, and has no timeout.
  The subprocess gives isolation and a hard `timeout=` kill. (It is **not** a full
  security sandbox — see gotchas.)
- **Why cap cost as well as iterations?** Iterations bound *count*; cost bounds
  *spend*. A model that emits huge solutions can blow the dollar budget well before
  the iteration cap. Real autonomous systems gate on both.
- **Why `temperature=0.2`, not 0?** Low enough to be near-deterministic for code,
  high enough to escape a repeated wrong fix on retry. (Temperature 0 is *low
  variance*, not deterministic — a Week 7 / Week 9 gotcha.)
- **Why AST before edit (Lab 3)?** Patching by string-matching breaks callers. The
  agent reads `read_ast_summary()` first so it edits with structural awareness, and
  `fix_file_in_repo` re-parses the LLM output with `ast.parse` to reject invalid
  Python *before* it ever touches the file.
- **Why restore-on-failure (Lab 4)?** A code agent must never leave the repo in a
  broken state. If the fix doesn't pass, `git_agent.py` rewrites the original source
  and commits nothing.

---

## Gotchas

- **The sandbox is isolation, not security.** Subprocess + timeout stops infinite
  loops and protects your process state, but the child still has filesystem and
  network access. For genuinely hostile code you need containers/seccomp/no-network
  — connect this to Week 7 Lab 5 (tool sandboxing, path-traversal allowlists).
- **The test suite *is* the spec.** The agent optimises for exactly what the tests
  assert. Weak tests → confidently-wrong code that passes. Garbage-in spec, garbage
  out.
- **Without a cost cap, a wrong-but-plausible fix can loop to the iteration limit**
  every run. Treat `max_cost_usd` as a production requirement, not a nicety.

## Dependencies

```bash
pip install gitpython pytest        # astroid optional, for richer AST work
```
`openai`, `python-dotenv`, and `pytest` come with the course environment.

## Setup

Uses the same `.env` as the rest of the course — add your `OPENAI_API_KEY`.
**No external services required**; everything runs locally in temp directories.

## Cost estimate

~$0.05–0.20 per bug-fix run (gpt-4o-mini for iteration; bump to gpt-4o for hard
cases). The budget cap defaults to `$0.50`.

---

## How this connects

- **← Week 1:** this is the hand-rolled agent loop again, now with a real reward
  function instead of an LLM judge.
- **← Week 7 Lab 5:** the security gap in `sandbox.py` is exactly what tool
  sandboxing addresses.
- **→ Week 9 (HITL):** code agents are *fully* autonomous — Week 9 adds the gate
  that keeps a human in control when the action is irreversible (e.g. push, deploy).

**Next:** [Week 9 Primer](../9_human_in_the_loop/PRIMER.md) — when full autonomy is
the wrong answer, and how to insert humans selectively.
