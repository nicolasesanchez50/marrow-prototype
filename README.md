# Marrow — recognize the user by how they move, catch the impostor

Marrow v0.1: an open-source, offline Python prototype that decides whether the
person at the computer is the enrolled user, from the ordinary input events the
operating system already sees (pointer movement + clicks + text as it arrives).

- No audio, no camera, no input-device assumption.
- Reads recordings in a plain, documented, OS-neutral JSONL format (ms
  timestamps, pointer positions as fractions of the screen, clicks, text events
  labeled by arrival: typed / onscreen / inserted).
- Ships with fake data only: two input styles (standard mouse+keyboard, slow
  pointer + dictation-inserted text), each with a tired owner, a stranger, and a
  mimic who copies the owner's habits.
- Passes the swap test: a profile trained on someone else does not open for the
  owner.
- Reports FAR/FRR at a threshold fixed before testing.

## Run

    python3 marrow.py demo

One command, offline, zero dependencies beyond the Python standard library
(numpy optional — falls back to pure Python if absent). Trains on owner
sessions, evaluates genuine vs impostor, runs the swap test, writes
`results/results.json`.

## License

MIT. Method credit: whoever builds it. (Marrow stays the funder's.)
