require 'nn'

-- This is a skeleton module: it does NOT implement the normalization.
--
-- It should only be used as an intermediate to be converted into a module
-- with a real implementation like:
--   nn.CrossMapNormalization (fbcunn)
--   inn.SpatialCrossResponseNormalization (imagine-nn)
local LRN, parent = torch.class(
  'nn.LocalResponseNormalization', 'nn.Identity'
)

function LRN:__init(size, kappa, alpha, beta)
  parent.__init(self)
  self.size  = size
  self.kappa = kappa
  self.alpha = alpha
  self.beta  = beta
end

function LRN:updateOutput(input)
  print("[WARNING] nn.LocalResponseNormalization: void implementation!")
  return parent.updateOutput(self, input)
end

function LRN:updateGradInput(input, gradOutput)
  print("[WARNING] nn.LocalResponseNormalization: void implementation!")
  return parent.updateGradInput(self, input, gradOutput)
end

local init = function(cuda)
  return function(size, kappa, alpha, beta)
    local lrn = nn.LocalResponseNormalization(size, kappa, alpha, beta)
    if cuda then
      lrn:cuda()
    end
    return lrn
  end
end

return init
