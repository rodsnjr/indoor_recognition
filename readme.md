# Indoor Recognition

Image recognition of indoor objects, using deep neural networks.

The datasets used are:
* [Door/Stairs Dataset](https://github.com/rodsnjr/gvc_dataset)
* [SUN Database](https://groups.csail.mit.edu/vision/SUN/)

The deep neural networks used are made with [TF Slim Models](https://github.com/tensorflow/models/tree/master/slim) package.

# Setting things up

The repository is packaged with a conda environment file. [You can install conda](https://conda.io/docs/user-guide/install/index.html), and use the following to create the virtual environment.
> conda env create -f environment.yml

And to activate it you can use:
> source activate indoor_recognition

The environment will also package the [Jupyter Notebook](http://jupyter.org/) module, and some helper extensions to it.
To use / edit the notebooks you can simply start jupyter notebook:
> jupyter notebook

And select the correct kernel to it (indoor_recognition) one.

There's also the need to setup the TF Slim Models repository by clonning it:
> git clone https://github.com/tensorflow/models/tree/master/slim

And changing the setup.cfg file with the correct directories for your project:
> [dependencies]
> tf_slim_models = '/home/user/Git/models/slim/'

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
