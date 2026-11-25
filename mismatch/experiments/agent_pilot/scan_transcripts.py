"""Scan the agent-pilot transcripts for anything outside an episode's own folder.

usage: python experiments/agent_pilot/scan_transcripts.py [results/agent_solutions]

Counts tool calls by name and lists every call whose input names a path or resource outside the
episode folder (its recorded cwd) or the agent's own Claude Code temp folders, or that reaches the
network, the grader or site-packages. Each listed call is shown with whether it succeeded, so a
reviewer can check the few hits by hand. Licence: Apache-2.0.
"""

import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OWN_TMP = re.compile(r"(/private)?/tmp/claude-\d+/[^/\" ]+/[0-9a-f-]+/(scratchpad|tasks)")
SUSPECT = re.compile(r"(\.\./|/Users/|~/|/etc/|/tmp/|/private/|mismatch-envs|claude-runs|_reference|"
                     r"eval/harness|\bcurl\b|\bwget\b|pip install|https?://|site-packages)")


def events(path):
    for line in path.read_text().splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg = ev.get("message") if isinstance(ev.get("message"), dict) else {}
        yield ev, msg.get("content") if isinstance(msg.get("content"), list) else []


def scan(base):
    names, hits = collections.Counter(), []
    for t in sorted(base.glob("*/*/r*/transcript.jsonl")):
        run, cwd, calls, errors = "/".join(t.parts[-4:-1]), None, {}, {}
        for ev, content in events(t):
            if ev.get("type") == "system" and ev.get("subtype") == "init":
                cwd = ev.get("cwd")
            for c in content:
                if c.get("type") == "tool_use":
                    names[c["name"]] += 1
                    s = json.dumps(c.get("input"))
                    s = s.replace(cwd, "<CWD>") if cwd else s
                    s = OWN_TMP.sub("<OWN_TMP>", s)
                    if SUSPECT.search(s):
                        calls[c["id"]] = (c["name"], sorted(set(SUSPECT.findall(s))), s[:200])
                elif c.get("type") == "tool_result" and c.get("tool_use_id") in calls:
                    errors[c["tool_use_id"]] = bool(c.get("is_error"))
        hits += [(run, *v, "error" if errors.get(k) else "ok") for k, v in calls.items()]
    return names, hits


if __name__ == "__main__":
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "experiments/results/agent_solutions"
    names, hits = scan(base)
    print("tool calls:", sum(names.values()), dict(names.most_common()))
    print("calls naming something outside the episode folder:", len(hits))
    for h in hits:
        print(" ", h[0], h[1], h[2], h[4], "|", h[3])
