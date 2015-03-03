require 'nn'

local init = function(cuda)
  return function(size, kappa, alpha, beta)
    return nn.LocalResponseNormalization(size, kappa, alpha, beta)
  end
end

return init
