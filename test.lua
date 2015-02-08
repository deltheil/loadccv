require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')

require 'nn'
require 'image'

local net   = torch.load('net.bin')
local input = torch.load('input.bin')
local mean  = torch.load('mean.bin')
local path  = assert(arg[1])
local img   = image.load(arg[1], 3, 'byte'):float()

assert(img:size(3) == input.width)
assert(img:size(2) == input.height)

mean = image.scale(mean, input.width, input.height)
img  = img - mean

local out = net:forward(img)
local _,ind = torch.sort(out, true)
-- top-5
for i=1,5 do
  print("[" .. ind[i] .. "] " .. out[ind[i]])
end
