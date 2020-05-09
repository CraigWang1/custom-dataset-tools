# labelimg_help.py
labelimg_help.py is a Python shortcut script for labelling images using [labelImg](https://github.com/tzutalin/labelImg).

<a href="https://raw.githubusercontent.com/CraigWang1/custom-dataset-tools/master/images/labelling"><img src="https://raw.githubusercontent.com/CraigWang1/custom-dataset-tools/master/images/labelling.gif" title="Labelling shortcut demo"/></a>

Instead of pressing (ctrl s) + (d) + (w), this shortcut simplifies it to just pressing (e), making labelling much more efficient and enjoyable.

#### Usage:
```
$ cd ../../path_to_/custom-dataset-tools/labelling
$ python3 labelimg_help.py

Shortcut activated. 
Workflow: Draw box, press 'e'. 
If annotating small images, press 'r' to zoom.
```

Once activated, go to back to labelImg and start annotating! Once you're done, click on your terminal and press `ctrl c` to kill the script and exit.

(The `e` key automatically saves the annotation, moves to the next image, and selects the rectangle tool to annotate another box.)

Extra tip: Use the default class so you don't have to retype it.
