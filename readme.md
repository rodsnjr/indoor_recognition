# Indoor Recognition

Image recognition of indoor objects, using deep neural networks.

The datasets used are:
* [Door/Stairs Dataset](https://github.com/rodsnjr/gvc_dataset)
* [SUN Database](https://groups.csail.mit.edu/vision/SUN/)

The deep neural networks used are made with [TF Slim](https://github.com/tensorflow/models/tree/master/slim) package.

# Repository Organization

This repository is organized with [Jupyter Notebooks](http://jupyter.org/), helper, datasets, and testing scripts.

The results and training/evaluation/etc are in the *.ipynb files, and the helpers scripts are in the *.py files.
The notebooks are named with the networks used for each of the 

```

indoor_recognition
│   README.md
│   vgg.ipynb (vgg network notebook)
│
└─  indoor_recognition
	│
	└─ datasets
	│   │	__init__.py
	│   │   sun.py
	│   │   gvc_indoor.py
	│   └─  gvc (dataset directory, will be generated/created with the scripts)
	│   │     *.jpg / *.npy / *.tfrecord (compressed files)
	│   │
	│   └─  sun (dataset directory)│
	│         *.jpg / *.npy / *.tfrecord (compressed files) 
	│
	└─ helpers
	│	__init__.py (python module file)
	│	*_helper.py (naming pattern)
	│
	└─ tests
		__init__.py (python module file)
		*_test.py (naming pattern)
    
```
