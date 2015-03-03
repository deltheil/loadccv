require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')

require 'nn'
require 'image'
require 'loadccv' -- to load the `nn.LocalResponseNormalization` dummy module

local meta  = torch.load('meta.bin')

if meta.cuda then
  require 'cunn'
end
require(meta.package) -- e.g 'cudnn'
require(meta.lrn)     -- e.g 'inn'

local net   = torch.load('net.bin')
local path  = assert(arg[1])
local img   = image.load(arg[1], 3, 'byte'):float()

assert(img:size(3) == meta.input.width)
assert(img:size(2) == meta.input.height)

if meta.mean then
  img = img - image.scale(meta.mean, meta.input.width, meta.input.height)
else
  print('[warn] no mean activity')
end

if meta.cuda then
 img = img:cuda():view(1, img:size(1), img:size(2), img:size(3))
end

local out = net:forward(img)

if meta.cuda then
  out = out:float():squeeze()
end

local _,ind = torch.sort(out, true)
-- top-5
for i=1,5 do
  print("[" .. ind[i] .. "] " .. out[ind[i]])
end
