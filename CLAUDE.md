# Jev Voice Agent

Side project. Voice control for the Mac: ElevenLabs
Scribe + on-device Whisper for ears, Jev (TypeSafe) for every decision, code for actions.
Read README.md first; docs/PLAN.md has the design, docs/VIDEO.md the demo script.

Rules
- Filesystem writes only through `jevagent/sandbox.py`; allowed roots are exactly
  `~/Desktop/Folder JEV` and `~/Desktop/Folder JEV copy`. Never widen them.
- Credentials: macOS Keychain service `jev-voice-agent` (TYPESAFE_API_KEY,
  ELEVENLABS_API_KEY). Never in files.
- Anything that touches mic, camera, screen or other apps runs from Terminal.app
  (`scripts/in_terminal.sh ...`), which holds those permissions; IDE shells do not.
- A change is done when the evals say so: `pytest`, `evals/text_eval.py` (all three
  sets), `evals/audio_eval.py`, and `live_eval.py` for anything the executor touches.
- Use the TypeSafe skill for Jev questions; docs mirrored locally in docs/typesafe/
  (gitignored).
