# CaImAn-Wrapper


CaImAn Wrapper is a package that calls CaImAn functions from: https://github.com/flatironinstitute/CaImAn. CaImAn Wrapper allows the user to execute
their recommended pipeline for 1-photon miniscope data processing with motion correction and CNMF-E. 

## Installation
### Installing CaImAn
Follow the installation instructions in https://github.com/flatironinstitute/CaImAn to set up CaImAn and its associated virtual 
environment. 

### Installing CaImAn Wrapper
Clone this repository using a command prompt: start>programs>anaconda3>anaconda prompt

```bash
git clone https://github.com/wmau/CaImAn-Wrapper
```

Your IDE will need this folder in its path before you are able to import its contents. Activate the caiman environment and run Spyder
from Anaconda Prompt:
```
activate caiman
spyder
```

In tools>PYTHONPATH manager, add the CaImAn-Wrapper folder to the path and restart Spyder. You should be able to import caiman_wrapper
from the Spyder IPython console.
```
import caiman_wrapper
```

There are better ways to do this installation, but this should work for now. 

## Running CaImAn Wrapper
Running caiman_wrapper will crop and spatially downsample (2x) your videos first and save those new videos to a user-defined directory. 
There are flags that allow you to skip these steps if you wish. Make sure your video files are labeled with leading 0s (e.g., 
movie-000.tif, movie-001.tif) for them to be sorted correctly (i.e., movie-10.tif will be sorted before movie-2.tif, which is not good). 
This module was originally written for analyzing Inscopix recordings, which omit the numeric tag on its first tif file, so there is a tag
(default True) to move that file to the front. Set is_inscopix=False to not do this.  

For default use: 
```
import caiman_wrapper as cw
raw_file_dir = 'directory//where//your//raw//files//are'
destination_dir = 'directory//where//you//want//your//analyses//to//go'
session = cw.CaimanWrapper(raw_file_dir, destination_dir)
```

Analyses run in two phases. The first phase 