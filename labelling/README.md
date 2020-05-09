# labelimg_help.py
labelimg_help.py is a Python shortcut script for labelling images using [labelImg](https://github.com/tzutalin/labelImg).

<a href="https://raw.githubusercontent.com/CraigWang1/custom-dataset-tools/master/images/labelling.gif">

Improves to 1 keystroke/image.

#### Usage:
```
$ cd ../../path_to_/custom-dataset-tools/labelling
$ python3 labelimg_help.py
```

Once activated, go to your labelImg window and draw your first rectangle. 

Workflow: draw box, press `e`, repeat.

You can also press the `r` key to zoom in if annotating small objects.

(The `e` key automatically saves the annotation, moves to the next image, and selects the rectangle tool to annotate another box.)

Once you're done, click on your terminal and press `ctrl c` to kill the script and exit.

Extra tip: Use the default class so you don't have to retype it.
