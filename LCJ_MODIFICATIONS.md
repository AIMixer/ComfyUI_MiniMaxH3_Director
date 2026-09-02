# LCJ Local Modifications

## 2026-08-31

- Replaced the shared “选已有” media picker list with direct thumbnail cards.
- Added `ComfyUI Input` and `ComfyUI Output` tabs to the picker.
- Output images are saved with ComfyUI’s `[output]` annotation so image loading remains correct.
- Director was updated from upstream commit `14c1f38` to `9007d9a` before this local UI change.

## 2026-08-31 (follow-up)

- Fixed the Input/Output thumbnail gallery height and forced vertical scrolling so all existing files remain selectable.
