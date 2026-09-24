# Jev Voice Agent

Talk to your Mac and it acts, often before you finish the sentence.

A replication of [Andy Gao's demo](https://youtube.com/shorts/82WPgjoRzBc) built on
[Jev](https://docs.typesafe.ai), TypeSafe's System One model. Jev never writes text: every
decision the agent makes (what you want, which app, which file, which words are the search
query, whether you were talking to the computer at all) is a typed question with a
probability. Code owns everything else.

```
mic ─┬─> ElevenLabs Scribe v2 Realtime ──> partial / committed transcripts ─┐
     └─> on-device Whisper (MLX, every 200 ms) ──> fast transcripts ────────┤
                                                                            v
                      Jev: one request per clause, ~25 questions in parallel
                      (tool Choice, app/site/file Choices, span Choices for free
                       text, "said to the computer?" Noul, "cut off?" Noul)
                                                                            v
                      policy (code): fire now / wait / carry to next segment
                                                                            v
                      executor: open, osascript, ffmpeg, screencapture, Sandbox
                                                                            v
                      ElevenLabs Flash TTS for short spoken replies (mic gated)
```

## What it can do

open / quit apps · create notes, set their title, write in them (Jev Notes window) ·
Google and YouTube search · open websites · close the tab · take a photo · take a
screenshot · create, rename, move, open, delete files and folders · list the folder ·
volume · dark mode · tell the time · stop

Deletes move items to a hidden `.trash/` inside the root, so nothing is erased for good.
Every create, rename, move and delete is confined to `~/Desktop/Folder JEV` or
`~/Desktop/Folder JEV copy` (the only allowed roots), enforced in `jevagent/sandbox.py`
and unit-tested against traversal and symlink escapes. Notes and photos are files in the
root. The agent never quits Terminal, VS Code, Claude, OBS or Finder. The Jev Notes page is
served on 127.0.0.1 only, checks the Host header and needs a random per-run key.

## Why it is fast

- **Two ears.** Scribe gives accurate transcripts but only about one partial per second.
  An on-device Whisper (base.en on the Apple GPU, ~50 ms per pass) re-transcribes the
  current utterance every 200 ms. The fast ear is trusted only for closed-vocabulary
  commands (apps, sites, volume, photo...), where a misheard letter still maps to the
  right option; free text (search queries, titles, names) waits for Scribe.
- **Speculative fan-out.** One Jev request asks every question for every tool at once
  (~300 ms); code reads only the answers the chosen tool needs. While Jev decides where a
  sentence splits into separate commands, it is already deciding every possible piece.
- **Early fire.** `open_app`, `open_website` and `show_folder` fire on a partial
  transcript when Jev is confident and the answer is stable.
- **Quiet fire.** Everything else fires ~300 ms after you stop talking (measured on the
  audio), once a transcript that covers your last words has arrived.

## Results (2026-09-24)

| Eval | Result |
|---|---|
| Text, tuning set (99 utterances) | 100%, 0 false actions |
| Text, held-out set 1 (70, written independently) | 100% on first run; later used for tuning, so no longer blind |
| Text, held-out set 2 (80, untouched, run once) | **98.8%**, 0 false actions |
| Audio: 4 spoken scenarios, 4 voices, real Scribe + fast lane | **27/27 actions, 0 false fires** |
| Open-app/site before the sentence ends | 5 of 8, median 0.5 s early |
| All actions vs end of speech | median +0.7 s |
| Live, real actions, `Folder JEV` and `Folder JEV copy` | **14/14 actions, 11/11 side-effect checks** (before the final security fixes; those are covered by unit tests and the audio eval, which runs file actions for real) |
| Idle room, 60 s of real microphone | 0 transcripts, 0 actions |

## Run it

```bash
# once
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
security add-generic-password -s jev-voice-agent -a TYPESAFE_API_KEY -w '<key>'
security add-generic-password -s jev-voice-agent -a ELEVENLABS_API_KEY -w '<key>'

# from Terminal.app (it needs Microphone, Camera, Accessibility, Screen Recording)
python -m jevagent run                         # live, root = ~/Desktop/Folder JEV
python -m jevagent run --root "~/Desktop/Folder JEV copy"
python -m jevagent run --mic "Samsung"         # any input device name substring
python -m jevagent run --dry-run --quiet       # decide, do nothing, no voice
python -m jevagent text "open chrome and search for italian recipes"
python -m jevagent devices
```

Say "goodbye" or press Ctrl+C to stop. Every run writes a timeline to `runs/`.

## Evals

```bash
python -m pytest                                      # sandbox, notes server, executor, text
python evals/text_eval.py [--cases evals/heldout2.jsonl]
python evals/audio_eval.py                            # ElevenLabs voices -> Scribe -> agent
scripts/in_terminal.sh python evals/live_eval.py      # real actions + side-effect checks
```

Known limit: `live_eval.py --loopback` (speakers into the built-in mic) only hears the
first seconds, because the MacBook's echo cancellation learns and removes its own speaker
output. A human voice is not affected, and the same property keeps the agent from
hearing its own replies.

## License

MIT, see LICENSE.

## Layout

```
jevagent/  brain.py (Jev)  agent.py (policy)  audio.py (Scribe, TTS, mic)  fastear.py
           executor.py  sandbox.py  notes_app.py  spec.py (tools)  text.py  __main__.py
evals/     cases.jsonl  heldout.jsonl  heldout2.jsonl  scenarios.json  *_eval.py
tests/     test_sandbox.py  test_text.py  test_notes_app.py  test_executor_notes.py
docs/      PLAN.md  reference-video.md  VIDEO.md
```
