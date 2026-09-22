# Reviewed development labels

These files document assistant visual review of local D2E-480p screenshots.
They are not human-expert annotations and do not establish player intent.

- `gameplay-curation-001.json` replaces one browser/YouTube clip with a gameplay
  clip from the same Grounded recording session.
- `menu-grounding-001.json` records whether a large menu overlay is visible in
  each of the 64 training and 64 separate-session validation frames. Current
  image SHA-256 hashes bind each label to its reviewed screenshot.

Source: `open-world-agents/D2E-480p`, revision
`f075f7e25df6f6d385840a836f86bf92dfb877ff`, CC BY-NC 4.0.
Source videos and images are not included here. These development labels and
synthetic conditional-control probes do not create a new untouched test set.

`laya_vision_stitch.grounding_data` derives questions and paired goal probes from
these labels only for offline training/evaluation. The runtime model never reads
these files and has no menu-state rule.
