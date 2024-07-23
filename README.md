# binfpy: A Python library for Bioinformatics
This is the public version of binfpy: A Python library for Bioinformatics. 
The library is primarily intended for teaching bioinformatics and scientific computing at UQ.

You are free to use this code for any purpose.
The code is changing continually and a number of people contribute, so we make no guarantees to it being bug free.

## Installation

Watch the installation demo [here](https://youtu.be/22F-_153DVw?si=wEywFzwhXHfwLqkj).

1) Clone the binfpy repository:

```
git clone https://github.com/bodenlab/binfpy.git
```

2) Create a conda environment:

```
conda create --name binfpy python=3.8
```

3) Activate the environment:

```
conda activate binfpy
```

4) Navigate into the binfpy directory. 

```
cd <path-to-download>/binfpy
```

5) Install the package with -e to allow for editing. 

```
pip install -e .
```

## Usage in a notebook

The clearest way to specify what library you want to use would look something like this.

```
import binfpy.webservice as webservice
```

Using `help` can also print out some useful information. You can also get information about specific functions. 
```
help(webservice)
help(webservice.getGODef)
```

It's also possible to only import certain functions or classes from the library. Here, I'll create an instance of a Sequence object.
```
from binfpy.sequence import Sequence

my_seq = Sequence("ATG", name="hello")
print(my_seq)
```

