require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')

require 'nn'
require 'image'

local net   = torch.load('net.bin')
local meta  = torch.load('meta.bin')
local path  = assert(arg[1])
local img   = image.load(arg[1], 3, 'byte'):float()

assert(img:size(3) == meta.input.width)
assert(img:size(2) == meta.input.height)

if meta.mean then
  img = img - image.scale(meta.mean, meta.input.width, meta.input.height)
else
  print('[warn] no mean activity')
end

local out = net:forward(img)
local _,ind = torch.sort(out, true)
-- top-5
for i=1,5 do
  print("[" .. ind[i] .. "] " .. out[ind[i]])
end
