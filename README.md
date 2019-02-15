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

### Running CaImAn Wrapper
