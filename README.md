# pi3-blender
Blender addon for Pi3 3D reconstruction

Input an image folder which contains single or multiple images, then you will get point could geometry nodes with material.

This blender addon is based on [Pi3](https://github.com/yyfz/Pi3). Be careful that vggt is under non-commercial license.

## Usage
1. Download Pi3 model from operation panel.
2. select an image folder or a mp4 video file.
3. Generate.



## Installation (only the first time)
1. Download Zip from this github repo.
2. Toggle System Console for installation logs tracking.
3. Install addon in blender preference with "Install from Disk" and select downloaded zip.
4. Wait for Pi3 git clone and python dependencies installation.
5. After addon activated, download Pi3 model from operation panel.



## Tested on
- Win11
- Blender 4.2
- cuda 12.6

## Notes
You can start with sample_images in this repository. "ramen" folder contains one single image, and "dog" folder contains four images.
