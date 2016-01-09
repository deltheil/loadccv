require 'nn'

local create_conv = function(ni, no, kw, kh, dw, dh, zp, weight, bias)
  local conv = nn.SpatialConvolutionMM(ni, no, kw, kh, dw, dh, zp)
  conv.weight:copy(weight)
  conv.bias:copy(bias)
  return conv
end

local create_relu = function()
  return nn.ReLU()
end

local create_max_pool = function(kw, kh, dw, dh)
  return nn.SpatialMaxPooling(kw, kh, dw, dh)
end

local create_avg_pool = function(kw, kh, dw, dh)
  return nn.SpatialAveragePooling(kw, kh, dw, dh)
end

local create_cross_map_lrn = function(size, kappa, alpha, beta)
  return nn.SpatialCrossMapLRN(size, alpha, beta, kappa)
end

return {
  cuda                       = function() return false end,
  SpatialConvolution         = create_conv,
  ReLU                       = create_relu,
  SpatialMaxPooling          = create_max_pool,
  SpatialAveragePooling      = create_avg_pool,
  SpatialCrossMapLRN         = create_cross_map_lrn,
}
