require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')

require 'nn'
require 'image'
require 'loadccv' -- to load the `nn.LocalResponseNormalization` dummy module

local pl = require('pl.import_into')()

local args = pl.lapp([[
Forward a test image in a network converted by loadccv
  -n,--net     (default 'net.bin')       path to the network created by loadccv
  -m,--meta    (default 'meta.bin')      path to the metadata created by loadccv
  -w,--words   (default 'imgnet.words')  path to ImageNet 2012 words mapping
  -v,--verbose                           be talkative
  <input>      (string)                  path to the test image
]])

for _,opt in pairs{'net', 'meta', 'input'} do
  assert(
    pl.path.isfile(args[opt]),
    opt .. ': `' .. args[opt] .. '` not found or invalid (see usage)'
  )
end

local words = pl.utils.readlines(args.words)
assert(
  #words == 1000,
  'expected 1,000 words (= classes), got ' .. #words
)

-------------------------------------------------
-- load extra dependencies
-------------------------------------------------

local ok, network, meta, img, w, h

ok, meta = pcall(torch.load, args.meta)
if not ok then error('could not load meta: ' .. meta) end

w, h = meta.input.width, meta.input.height

if meta.cuda then
  require 'cunn'
end
require(meta.backend) -- e.g 'cudnn'

if args.verbose then
  print(
    string.format("%-20s: %s", 'CUDA', meta.cuda and 'enabled' or 'disabled')
  )
  print(
    string.format("%-20s: %s", 'backend', meta.backend)
  )
  print(
    string.format("%-20s: %dx%d", 'expected image size', w, h)
  )
  print(
    string.format("%-20s: %s", 'mean image', meta.mean and 'yes' or 'no')
  )
end

-------------------------------------------------
-- load inputs
-------------------------------------------------

ok, network = pcall(torch.load, args.net)
if not ok then error('could not load network: ' .. network) end

ok, img = pcall(image.load, args.input, 3, 'byte')
if not ok then error('could not load image: ' .. img) end

-------------------------------------------------
-- convert input image into the expectes format
-------------------------------------------------

img = img:float()
img = image.scale(img, w, h)
if meta.mean then
  img = img - image.scale(meta.mean, w, h)
end
if meta.cuda then
  img = img:cuda():view(1, img:size(1), img:size(2), img:size(3))
end

-------------------------------------------------
-- forward and print results
-------------------------------------------------
local outputs = network:forward(img)

if meta.cuda then
  outputs = outputs:float():squeeze()
end

local _,ind = torch.sort(outputs, true)

print('\nRESULTS (top-5):')
print('----------------')

-- top-5
for i=1,5 do
  print(string.format("score = %f: %s", outputs[ind[i]], words[ind[i]]))
end
