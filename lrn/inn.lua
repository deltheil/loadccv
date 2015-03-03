require 'inn'

local init = function(cuda)
  assert(cuda, 'inn SpatialCrossResponseNormalization has no CPU support')
  return function(size, kappa, alpha, beta)
    return inn.SpatialCrossResponseNormalization(size, alpha, beta, kappa)
  end
end

return init
