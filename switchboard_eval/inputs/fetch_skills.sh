#!/usr/bin/env bash
# Re-clone the in-scope skill repos so harvest_examples.py can regenerate examples.jsonl.
# Out of scope by decision: skill-communications (WIP), skill-plex (MA covers Plex audio; no video use),
# caffeinewiz and stocks (disabled), openHAB (not used), skill-messaging (comms.* dropped).
set -euo pipefail
DIR="${1:-skills/repos}"; mkdir -p "$DIR"; cd "$DIR"
while read -r repo; do
  [ -z "$repo" ] && continue
  dest="$(echo "$repo" | tr / _)"
  [ -d "$dest" ] && { echo "skip $repo"; continue; }
  git clone -q --depth 1 "https://github.com/$repo" "$dest" && echo "ok   $repo" || echo "FAIL $repo"
done <<'LIST'
NeonGeckoCom/skill-about
NeonGeckoCom/skill-alerts
NeonGeckoCom/skill-data_controls
NeonGeckoCom/skill-date_time
NeonGeckoCom/skill-demo
NeonGeckoCom/skill-device_controls
NeonGeckoCom/skill-ip_address
NeonGeckoCom/skill-jokes
NeonGeckoCom/skill-news
NeonGeckoCom/skill-personal
NeonGeckoCom/skill-speak
NeonGeckoCom/skill-spelling
NeonGeckoCom/skill-support_helper
NeonGeckoCom/skill-translation
NeonGeckoCom/skill-update
NeonGeckoCom/skill-user_settings
NeonGeckoCom/skill-weather
NeonGeckoCom/skill-wikipedia
OpenVoiceOS/ovos-skill-easter-eggs
OpenVoiceOS/ovos-skill-randomness
OpenVoiceOS/ovos-skill-volume
OscillateLabsLLC/ovos-skill-spelling
OscillateLabsLLC/skill-homeassistant
OscillateLabsLLC/skill-mac-application-launcher
OscillateLabsLLC/skill-meal-plan
OscillateLabsLLC/skill-monster-scanner
OscillateLabsLLC/skill-musicassistant
LIST
