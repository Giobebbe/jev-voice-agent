# Recording the YouTube demo

## Before recording
1. Terminal.app has Microphone, Camera, Accessibility, Screen & System Audio Recording.
2. Plug in the Samsung mic, run `python -m jevagent devices` and pass its name to `--mic`.
3. Close Apple Notes and any window you don't want on camera. The agent writes notes in
   its own "Jev Notes" window, so your real notes never show.
4. Turn on the room light (the photo action uses the MacBook camera directly).
5. Start `python -m jevagent run --mic "Samsung"` in a Terminal window you keep visible.
   It shows the live transcript and every action with its timing
   (`⚡ ... EARLY` = fired before you finished the sentence).

## Run of show (every line below passed the evals)
1. "All right, can you open up the notes app for me?"
2. "And once you're there, can you create a new note?"
3. "And inside this new note, let's make the title say, hello."
4. "Okay, let's move on and can you open up the Arc browser?"   (opens Chrome)
5. "And once you're there, can you Google search Norbert Wiener?"
6. "Now can you open up X dot com?"
7. "Okay, now can you open up the photo booth?"
8. "And let's take a picture of me."   (photo saved in Folder JEV/Photos, shown in Preview)

Extras that look good on camera:
- "Open the notes app and create a new note for my groceries." (the window opens mid-sentence)
- "Create a folder called Project Apollo." / "Rename Project Apollo to Project Artemis." /
  "What's in the folder?" (spoken answer) / "Delete Project Artemis."
- "Search YouTube for Andrej Karpathy." / "Close this tab."
- "Take a screenshot." / "What time is it?" / "Goodbye."
- Talk to the camera ("so guys, I usually open Chrome first") and show that nothing happens.

## Tips
- Speak normally. Pausing mid-command is fine: the agent waits about 2.5 s for the rest.
- Chit-chat, reactions and narration are ignored by design.
- An app or site opens as soon as its name is clear; commands with free text (a search, a
  title, a file name) run about half a second after you stop talking.
