# Input Images

This folder contains all input images organized by category:

- `walls/` - Room images with visible walls for product placement
- `products/` - Product images to be placed on walls
  - `tvs/` - Television images (preferably with transparent backgrounds)
  - `paintings/` - Painting and artwork images
  - `frames/` - Photo frames and similar items

## Usage

Place your room images in the `walls/` folder and your product images in the appropriate `products/` subfolder.

## Example Commands

```bash
# Place a TV on a living room wall
python wall_ar_pipeline.py --room input/walls/living_room.jpg --product input/products/tvs/modern_tv.png --type tv --output output/results/living_room_with_tv.png

# Add a painting to a bedroom
python wall_ar_pipeline.py --room input/walls/bedroom.jpg --product input/products/paintings/abstract_art.png --type painting --output output/results/bedroom_with_art.png

# Place a photo frame in an office
python wall_ar_pipeline.py --room input/walls/office.jpg --product input/products/frames/family_photo.png --type frame --output output/results/office_with_frame.png
```
