# Sudoku Vision

## Description:

Sudoku Vision is a CLI Python script that extracts the data from images containing sudoku puzzles using OpenCV. It was written for the Concepts and Applications in Computer Vision course at University of Bucharest, Faculty of Mathematics and Informatics. It's not perfect and it's very simplistic. It also doesn't output what number each cell contains, only if a cell is empty or not.

## Dependencies:

Python        >= 3.9.5
numpy         >= 1.19.5
opencv-python >= 4.5.4.60

This is what I used when writing it, but you can probably get it to work on older versions without too many problems. OpenCV should at least be 4.5.4, though.

## Usage examples:

Since it's packed tightly in a single file, you can use the same script to run both classic and jigsaw puzzles through it, selecting the mode through ``-m / --mode=`` option.

Make sure script is executable before running: ``chmod +x sudokuVision.py`` (or you could just pass it to an interpreter directly)


To run classic sudoku puzzles:

``$ ./sudokuVision.py -m classic -t /path/to/truth/dir -o /path/to/output/dir --check --verbose image1.jpg image2.jpg image3.jpg``


To run jigsaw sudoku puzzles:

``$ ./sudokuVision.py -m jigsaw -t /path/to/truth/dir -o /path/to/output/dir --check --verbose image1.jpg image2.jpg image3.jpg``


Passing the files dynamically through arguments has some advantages. One of them is that you can make use of shell globbing:

``$ ./sudokuVision.py -m classic ./train/classic/*.jpg``

The above command will run Sudoku Vision on all the .jpg files in ``./train/classic/``.

You can also check the usage by typing in:

``$ ./sudokuVision.py --help``

This will print the following information:

```
Usage: sudokuVision [OPTION]... FILE
Extract sudoku from FILE and output data in a .txt file

Options:
  -h, --help                 Display this help and exit
  -v, --verbose              Print more info on the steps
  -c, --check                Check solution against ground truths and count correct guesses

  -m, --mode=<MODE>          Mode to run on [classic, jigsaw]
  -t, --truth-path=<PATH>    Path to directory containing truth files
                             (if not provided, it looks in the same path as the files to run on)
  -o, --output-path=<PATH>   Path to directory to write solutions in
                             (if not provided, it creates an output dir in the current dir)

Written by Costinteo for Concepts and Applications in Computer Vision course.
University of Bucharest, Faculty of Mathematics and Informatics.
Licensed under GPL v3. For more information, access: <https://github.com/Costinteo/>
```
