# data/
This directory contains the processed dataset. Images are sorted into directories with format "MANUFACTURER_FAMILY" (e.g. Airbus_A300). All variants (e.g. -200, -300) are included in the same folder.

## Build
To create the processed dataset, the ```unpackDatset.py``` script may be used. It builds a lookup table of the manufacturer and family of aircraft per unique id and sorts the images into the appropriate directories. The script also removes any metadata from the bottom of the image.

**IMPORTANT NOTE:**
Due to the size of the dataset, only a subset is included in this remote repository.