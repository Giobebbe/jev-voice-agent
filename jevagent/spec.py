"""What the agent can do, written for Jev.

Each tool has a description (an option of the `tool` Choice) and the arguments it reads.
Arguments are shared questions: every request asks all of them speculatively and the
chosen tool reads only its own (docs/typesafe/patterns_fan-out.md).
"""

from __future__ import annotations

from dataclasses import dataclass

NONE = "none"


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args: tuple[str, ...] = ()
    optional: tuple[str, ...] = ()
    early: bool = False  # may fire on a partial transcript
    final_only: bool = False  # only on a committed transcript (destructive)


TOOLS: dict[str, Tool] = {t.name: t for t in [
    Tool(NONE, "No request for the computer: small talk, thanks, praise, reactions like 'great' or 'nice', "
               "filler, talking to someone else, or a fragment that does not yet say what to do"),
    Tool("open_app", "Open, launch or switch to an application installed on this Mac (a web browser like Chrome, "
                     "Safari or Arc, the notes app, Photo Booth, TextEdit, Finder, Calculator...)", ("app",), early=True),
    Tool("quit_app", "Close, quit or exit an application", ("app",)),
    Tool("create_note", "Create a new note in the notes app", optional=("title",)),
    Tool("set_note_title", "Give the current note a title, or change what its title says", ("title",)),
    Tool("write_in_note", "Write, add or type some text into the current note", ("note_text",)),
    Tool("web_search", "Search the web or Google for something", ("query",)),
    Tool("youtube_search", "Search YouTube for videos about something", ("query",)),
    Tool("open_website", "Open or go to a website or online service, e.g. X (Twitter), YouTube, Gmail, GitHub, "
                         "Wikipedia, or any web address", ("site",), early=True),
    Tool("close_tab", "Close the current browser tab"),
    Tool("take_photo", "Take a picture or a selfie of the user with the camera"),
    Tool("take_screenshot", "Take a screenshot or screen capture of the screen"),
    Tool("create_folder", "Create or make a new folder", ("name",)),
    Tool("create_file", "Create or make a new file that is not a note, e.g. a text file or markdown file", ("name",)),
    Tool("delete_item", "Delete, remove or trash an existing file or folder", ("item",), final_only=True),
    Tool("rename_item", "Rename an existing file or folder", ("item", "name")),
    Tool("move_item", "Move an existing file or folder into another folder", ("item", "dest")),
    Tool("open_item", "Open one of the existing files, or one of the subfolders listed in `files`", ("item",)),
    Tool("show_folder", "Show or open the main working folder itself, the JEV folder, in Finder", early=True),
    Tool("list_items", "Say what files and folders are in the working folder"),
    Tool("set_volume", "Change the sound volume: louder, quieter, mute or unmute", ("volume",)),
    Tool("dark_mode", "Turn dark mode on or off", ("mode",)),
    Tool("tell_time", "Say the current time or date"),
    Tool("stop_listening", "Stop the assistant: goodbye, stop listening, shut down the assistant"),
]}

TOOL_QUESTION = "What does the user ask the computer to do in `command`?"

# Common app descriptions help the literal model map what people say to the app name.
APP_HINTS: dict[str, str] = {
    "Google Chrome": "Google Chrome, the web browser; also when the user says 'the browser', 'Chrome', "
                     "or names a browser that is not installed such as Arc",
    "Safari": "Safari web browser, only when the user says Safari",
    "Photo Booth": "Photo Booth, the camera and selfie app",
    "Jev Notes": "the notes app, where notes are written; use it whenever the user says 'the notes app', "
                 "'notes' or 'my notes' without saying Apple",
    "TextEdit": "TextEdit, the plain text editor",
    "Notes": "Apple Notes, only when the user explicitly says 'Apple Notes'",
    "Finder": "Finder, the file browser",
    "System Settings": "System Settings or System Preferences",
    "Calculator": "Calculator",
    "Calendar": "Calendar",
    "Music": "Apple Music player",
    "Preview": "Preview, the image and PDF viewer",
    "QuickTime Player": "QuickTime Player, video player",
    "FaceTime": "FaceTime video calls",
    "Messages": "Messages, iMessage",
    "Maps": "Apple Maps",
    "Weather": "Weather",
    "Clock": "Clock, alarms and timers",
    "Reminders": "Reminders",
    "Activity Monitor": "Activity Monitor",
    "App Store": "App Store",
    "WhatsApp": "WhatsApp",
    "Telegram": "Telegram",
    "Slack": "Slack",
    "Notion": "Notion",
    "Obsidian": "Obsidian",
    "Linear": "Linear",
    "Spotify": "Spotify",
}

SITES: dict[str, str] = {
    "x.com": "X, formerly Twitter (x.com)",
    "youtube.com": "YouTube",
    "google.com": "Google home page",
    "gmail.com": "Gmail, Google mail",
    "github.com": "GitHub",
    "linkedin.com": "LinkedIn",
    "reddit.com": "Reddit",
    "wikipedia.org": "Wikipedia",
    "typesafe.ai": "TypeSafe, the company that makes Jev",
    "elevenlabs.io": "ElevenLabs",
    "claude.ai": "Claude by Anthropic",
    "chatgpt.com": "ChatGPT",
    "amazon.com": "Amazon",
    "netflix.com": "Netflix",
    "instagram.com": "Instagram",
    "facebook.com": "Facebook",
    "skool.com": "Skool",
    "maps.google.com": "Google Maps",
}

VOLUME = {
    "up": "louder, turn the volume up",
    "down": "quieter, turn the volume down",
    "mute": "mute or silence the sound",
    "unmute": "unmute, bring the sound back",
    "max": "maximum volume",
    "half": "half or medium volume",
}

MODE = {"on": "turn dark mode on", "off": "turn dark mode off, back to light mode", "toggle": "switch dark mode"}

ABSENT = "__not_said__"
NOT_INSTALLED = "__not_installed__"

# Span questions: the answer is one of the spans of `command` (or ABSENT).
SPAN_QUESTIONS = {
    "title": "Which words from `command` does the user want as the title of the note?",
    "note_text": "Which words from `command` does the user want written into the note?",
    "query": "Which words from `command` are the thing the user wants to search for?",
    "name": "Which words from `command` are the name the user wants for the new or renamed file or folder?",
}
