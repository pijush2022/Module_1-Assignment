# Output Images

This folder contains processed results from the Wall AR Pipeline.

## Structure:
- `results/` - Final processed images with products placed on walls
- `debug/` - Debug visualizations and intermediate results (when enabled)

## File Naming:
Results are typically named based on the input files or timestamps.
Debug folders contain detailed processing information and visualizations.

## Debug Information:
When `--debug` flag is used or `save_debug=True` in code, additional files are created:
- `segmentation.png` - Wall segmentation visualization
- `placement.png` - Product placement preview
- `metadata.json` - Processing statistics and confidence scores

## Example Output Files:
- `living_room_with_tv.png` - Final result image
- `living_room_with_tv_debug/` - Debug folder with intermediate steps
  - `segmentation.png` - Shows detected wall regions
  - `placement.png` - Shows product positioning
  - `metadata.json` - Processing details and scores
