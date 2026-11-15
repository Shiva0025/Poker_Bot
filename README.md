# Pokerbots Starter Kit: Discard Hold'em

| File | What it is |
|---|---|
| `bot.py` | Starter bot (Protocol Specification §11). Replace its strategy with yours. |
| `engine.py` | Local match runner. Plays a full match between two bots under the official rules. |

Read the Problem Statement (game rules), the Protocol Specification (messages) and the
Rulebook (tournament, limits, conduct) before you start.

## Requirements

Python 3.11. The runner uses only the standard library; your bot may also use numpy and scipy.

## Quick start

    python3.11 engine.py bot.py bot.py

This plays a 400-hand Stage 1 match and prints a summary. A match is played in two legs of
200 hands. Each leg starts fresh bot processes and plays every deal once; the second leg
replays the same deals with the hole cards swapped. Your bot receives `init` at the start of
each leg and `match_end` at the end of each leg, and must exit after `match_end`.

## Options

| Option | Meaning |
|---|---|
| `--stage 2` | Stage 2 match: 1000 hands in two legs of 500; 75 s time bank and 225 s wall-clock per leg |
| `--hands N` | Play N hands in total over both legs (must be even) |
| `--seed S` | Fixed deal seed, so the same seed replays the same cards |
| `--verbose` | Print every hand to the console |
| `--log-dir DIR` | Where logs are written (default `logs`; `''` disables) |
| `--python PATH` | Interpreter used to run the bots |
| `--no-time-limits` | Turn off all clocks, for debugging only. Never judge timing with this on. |

## Logs

- `logs/match.log`: one line per hand, prefixed with the leg (`L1`, `L2`): hole cards, board,
  discards, actions (`x` check, `c6` call to 6, `r20` raise to 20, `f` fold) and the result.
- `logs/A_leg1_stderr.txt`, `A_leg2_stderr.txt`, `B_leg1_stderr.txt`, `B_leg2_stderr.txt`: each
  bot's stderr per leg (first 64 KB). Print your debug output with `print(..., file=sys.stderr)`.
  Anything else on stdout breaks the protocol.

## What the runner enforces (same as the official engine)

- Two legs per match with fresh bot processes; every deal once per leg, seats swapped in leg 2
- Duplicate deals, alternating seats, 200 chips per hand, blinds 1 and 2
- Minimum raises, all-in amounts and the 4-bet cap per street
- The simultaneous, exposed discard after the flop (also when all-in)
- Corrections: illegal or malformed actions become check, else fold; illegal discards become the
  lowest-ranked card (ties: spades, hearts, diamonds, clubs). A `notice` is sent each time.
- Time bank per leg, 5 s per reply, 5 s start-up per leg (not charged), wall-clock kill per leg.
  Exceeding any of them, crashing or closing stdout forfeits the whole match.

## Before you upload

- [ ] `python3.11 bot.py` runs with only the standard library, numpy and scipy
- [ ] Nothing but protocol JSON goes to stdout, and every reply is flushed
- [ ] The bot answers both `action_request` and `discard_request`
- [ ] The bot ignores message types and fields it does not know
- [ ] The bot exits after `match_end`
- [ ] A full 400-hand match (two legs) runs inside the per-leg time bank with zero notices:
      `python3.11 engine.py bot.py bot.py`
- [ ] Also try `--stage 2` to check your timing over 1000 hands

You may reuse any code from this kit, including the hand evaluator in `engine.py`. Material
supplied by the organisers is allowed under Rulebook 10.1.

---

# Team notes: our bot, tests and submission

**What to upload:** only `Starter_Kit/bot.py` (single file, ~166 KB, stdlib only; RB 5.10 limit is 5 MB).
No ZIP is needed: the bot reads no data files (its preflop table is embedded).

| Path | Purpose |
|---|---|
| `bot.py` | **The submission.** |
| `AUDIT.md` | Requirements checklist with rule references, audit findings, fixes, limitations |
| `TEST_RESULTS.md` | Commands, environment, seeds, measured results and their limits |
| `STRATEGY.md` | Plain-language description of the strategy for code review |
| `baseline/bot_original.py` | Unchanged pre-audit bot (SHA-256 `5243a40a…8fe48`) |
| `engine.py` | Local match runner |
| `dev/opponents/` | Our test opponents (style bots, variants of the original bot); never submitted |
| `dev/tools/` | Tests, benchmark, table generator; never submitted |
| `dev/results/` | Raw benchmark data (JSON lines, one per match) |

There is no local Python 3.11 on the development machine. Everything below runs in the official-version image
`python:3.11-slim` (Docker) from the `Starter_Kit` directory:

    # one full Stage 1 / Stage 2 match, clocks on, 1 vCPU, 1 GB
    docker run --rm --cpus 1 --memory 1g --network none -v "$PWD":/w -w /w python:3.11-slim \
        python engine.py bot.py bot.py --log-dir ''
    docker run --rm --cpus 1 --memory 1g --network none -v "$PWD":/w -w /w python:3.11-slim \
        python engine.py bot.py bot.py --stage 2 --log-dir ''

    # tests (each exits non-zero on failure)
    python3 dev/tools/test_protocol.py bot.py        # stdin/stdout protocol conformance
    python3 dev/tools/test_units.py bot.py           # regressions + legality fuzz
    python3 dev/tools/test_evaluator.py bot.py       # evaluator vs engine.best_hand
    python3 dev/tools/validate_pf.py bot.py          # preflop table checks + independent regeneration

    # benchmark (parallel, unchanged engine, clocks on) and report
    python3 dev/tools/bench.py --bots bot.py baseline/bot_original.py --opps orig tag value \
        --stage 2 --seeds 1-10 --jobs 8 --out dev/results/my_run.jsonl
    python3 dev/tools/report.py dev/results/my_run.jsonl --pair bot.py baseline/bot_original.py

    # regenerate the embedded preflop table (≈ 25 CPU-minutes at 16000 samples, 3 processes)
    python3 dev/tools/gen_pf.py bot.py 16000 3 dev/results/pf_blob.txt

    # final validation of the exact submission file in a clean directory
    mkdir -p dev/release && docker run --rm --cpus 1 --memory 1g --network none \
        -v "$PWD":/w:ro -v "$PWD/dev/release":/rel -w /w python:3.11-slim \
        python dev/tools/final_check.py /rel
