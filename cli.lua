local ccv   = require 'loadccv'
local pl    = require('pl.import_into')()

local args = pl.lapp([[
Load a ccv (libccv.org) network in Torch7.
  -o        (default '.')    output directory
  --softmax                  append a soft-max module to the network
  --spatial                  use spatial convolutions for fully-connected layers
  --backend (default 'nn')   specific backend for operations (nn | cunn | cudnn)
  --verbose                  print layers information
  <path>    (string)         path of the ccv network (sqlite3 file)
]])

assert(pl.path.isdir(args.o), args.o .. ' is not a directory')

local opts = {
  spatial = args.spatial,
  backend = args.backend,
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

torch.save(pl.path.join(args.o, 'net.bin'), net)
torch.save(pl.path.join(args.o, 'meta.bin'), meta)

print('Done. See: ' .. pl.stringx.join(', ', pl.dir.getfiles(args.o, '.bin')))
