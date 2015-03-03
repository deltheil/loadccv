local ccv   = require 'loadccv'
local lapp  = require 'pl.lapp'
local pathx = require 'pl.path'
local dirx  = require 'pl.dir'
local strx  = require 'pl.stringx'

local args = lapp([[
Load a ccv (libccv.org) network in Torch7.
  -o        (default '.')    output directory
  --softmax                  append a soft-max module to the network
  --spatial                  use spatial convolutions for fully-connected layers
  --package (default 'nn')   specific package for operations (nn | cunn | cudnn)
  --lrn     (default 'nn')   package for Local Response Norm. (nn | inn)
  --verbose                  print layers information
  <path>    (string)         path of the ccv network (sqlite3 file)
]])

assert(pathx.isdir(args.o), args.o .. ' is not a directory')

local opts = {
  spatial = args.spatial,
  package = args.package,
  lrn     = args.lrn,
  verbose = args.verbose,
}

local net, meta = ccv.load(args.path, opts)

if args.softmax then
  if args.spatial then
    net:add(nn.Reshape(meta.num_output))
  end
  net:add(nn.SoftMax())
end

if meta.cuda then
  net:cuda()
end

torch.save(pathx.join(args.o, 'net.bin'), net)
torch.save(pathx.join(args.o, 'meta.bin'), meta)

print('Done. See: ' .. strx.join(', ', dirx.getfiles(args.o, '.bin')))
