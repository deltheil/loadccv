local cudnn = require 'cudnn'

local init = function(cuda)
  assert(cuda, 'cudnn SpatialCrossMapLRN has no CPU support')
  assert(cudnn.version >= 3000, 'cudnn v3 or higher is required')
  return function(size, kappa, alpha, beta)
    return cudnn.SpatialCrossMapLRN(size, alpha, beta, kappa)
  end
end

return init
