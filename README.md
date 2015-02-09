# loadccv

A tool that lets you load [ccv](http://libccv.org/) networks in [Torch7](http://torch.ch/)
(inspired by [loadcaffe](https://github.com/szagoruyko/loadcaffe)).

Installing ccv is not required since this tools mimicks
[`ccv_convnet_read`](http://libccv.org/lib/ccv-convnet/#ccvconvnetread), i.e it
reads a ccv `.sqlite3` archive and builds the corresponding network with Torch
modules.

## Install

    luarocks make

## Usage

You can convert a ccv network right from the command-line:

```bash
loadccv /path/to/ccv/samples/image-net-2012-vgg-d.sqlite3
```

Or use the library in your program:

```Lua
local ccv = require 'loadccv'

local net = ccv.load('/path/to/ccv/samples/image-net-2012-vgg-d.sqlite3')
```
