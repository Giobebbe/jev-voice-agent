# Jev Voice Agent -- plan

## Goal
Replicate Andy Gao's demo (docs/reference-video.md): talk to the Mac, the Mac acts, and
acts **before the sentence is finished** when the intent is already clear. Jev (TypeSafe)
is the decision layer; our code does everything else. Validate with evals, not by eye.

## Architecture

```
mic (16 kHz PCM, 100 ms chunks)
  -> ElevenLabs Scribe v2 Realtime (WebSocket, VAD commits)
       partial transcripts every few hundred ms, committed transcript on silence
  -> Agent (latest-partial-wins worker)
       1. split the segment into sentences (code) and candidate boundaries ("and", "then")
       2. Jev: one Noul per boundary -> is this a new request?        (only if boundaries exist)
       3. Jev: one fan-out request per clause -> tool Choice + every argument question
              (app Choice, item Choice, site Choice, span Choices over n-grams of the clause,
               "finished?" Noul) -- speculative, all in parallel inside one call
       4. policy (code): fire now / wait / ask
            open_app, open_website  -> fire on a PARTIAL when tool+arg are confident
            everything else         -> fire when the clause is closed (next clause started,
                                       VAD commit, or 350 ms of quiet + finished Noul)
            delete                  -> only on a committed transcript
  -> Executor (sequential action queue, macOS open/osascript/ffmpeg/screencapture)
       every create/delete/rename/move goes through Sandbox (root = Folder JEV or Folder JEV copy)
  -> Speaker (ElevenLabs Flash TTS, cached per phrase; mic gated while speaking)
```

## Hard constraints (from Gio, 2026-09-24)
- Every filesystem create/delete/rename/move happens only inside `~/Desktop/Folder JEV`
  (sandbox phase) or `~/Desktop/Folder JEV copy` (real-run phase). Enforced in code
  (`Sandbox`), allowlisted roots, path traversal and symlink escape rejected, unit-tested.
- So notes are text files in the root opened in TextEdit (not iCloud Notes), photos are
  captured from the camera straight into the root (not the Photo Booth library).
- The agent never quits its own host apps (Terminal, VS Code, Claude, OBS).
- Credentials live in the macOS Keychain (service `jev-voice-agent`), never in files.

## Evals (the definition of done)
1. `evals/text_eval.py` -- ~90 typed utterances (video script, paraphrases, chit-chat
   negatives, partial prefixes) -> expected tool + args. Metrics: exact-match accuracy,
   false-action rate on negatives, Jev latency p50/p95.
2. `evals/audio_eval.py` -- the same scenarios spoken by ElevenLabs voices, streamed at
   real-time pace through the real Scribe WebSocket and the real agent (dry-run executor).
   Metrics: correct action sequence, decision time relative to end of speech (negative =
   before the user finished), early-fire rate on open_* commands.
3. `evals/live_eval.py` -- real executor in the sandbox root: replays the video scenario
   and file scenarios, then verifies side effects (files, TextEdit text, Chrome URL,
   photo bytes, running apps).
4. Acoustic loopback: the scenario is played on the Mac speakers and heard by the Mac
   microphone, full real pipeline.
Targets: >= 95% text accuracy, 0 false actions on chit-chat, open_* fired before the end
of the sentence in most cases, end-to-end action < 1.5 s after the words are spoken.
