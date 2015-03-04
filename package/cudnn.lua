require 'cudnn'

local create_conv = function(ni, no, kw, kh, dw, dh, zp, weight, bias)
  local padw, padh = zp, zp
  local conv = cudnn.SpatialConvolution(ni, no, kw, kh, dw, dh, padw, padh)
  conv.weight:copy(weight)
  conv.bias:copy(bias)
  return conv
end

local create_relu = function()
  return cudnn.ReLU(true)
end

local create_max_pool = function(kw, kh, dw, dh)
  return cudnn.SpatialMaxPooling(kw, kh, dw, dh)
end

local create_avg_pool = function(kw, kh, dw, dh)
  return cudnn.SpatialAveragePooling(kw, kh, dw, dh)
end

return {
  cuda                       = function() return true end,
  SpatialConvolution         = create_conv,
  ReLU                       = create_relu,
  SpatialMaxPooling          = create_max_pool,
  SpatialAveragePooling      = create_avg_pool,
}
