require 'torch'
torch.setdefaulttensortype('torch.FloatTensor')

require 'nn'

local sqlite3 = require 'lsqlite3'

local ffi = require 'ffi'
local C = ffi.load(package.searchpath('loadccv', package.cpath))
ffi.cdef 'void loadccv_half_precision_to_float(uint16_t*, float*, size_t);'

local CCV_TYPES = {
  'CCV_CONVNET_CONVOLUTIONAL',
  'CCV_CONVNET_FULL_CONNECT',
  'CCV_CONVNET_MAX_POOL',
  'CCV_CONVNET_AVERAGE_POOL',
  'CCV_CONVNET_LOCAL_RESPONSE_NORM',
}

local noop = function() end

local half_prec_to_float = function(str)
  local n = #str/ffi.sizeof('uint16_t')
  local h = ffi.new('uint16_t[?]', n); ffi.copy(h, str, #str)
  local f = ffi.new('float[?]', n)
  C.loadccv_half_precision_to_float(h, f, n)
  return ffi.string(f, n*ffi.sizeof('float'))
end

local copy_float_str = function(str, tensor) -- src, dst
  assert(tensor:isContiguous())
  assert(tensor:nElement() == #str/ffi.sizeof('float'))
  ffi.copy(tensor:data(), str, #str)
end

local fetch_convnet_input_size = function(db)
  local stmt = db:prepare [[
    SELECT
    input_height,
    input_width
    FROM convnet_params
    WHERE convnet = 0
]]
  local width, height
  local rc = stmt:step()
  if rc == sqlite3.ROW then
    local res = stmt:get_named_values()
    width, height = res.input_width, res.input_height
  else
    error('could not read input size (code=' .. rc .. ')')
  end
  stmt:finalize()
  return width, height
end

local fetch_all_layers = function(db)
  return db:nrows [[
    SELECT
    layer as id,
    type,
    input_matrix_rows,
    input_matrix_cols,
    input_matrix_channels,
    input_matrix_partition,
    input_node_count,
    output_rows,
    output_cols,
    output_channels,
    output_partition,
    output_count,
    output_strides,
    output_border,
    output_size,
    output_kappa,
    output_alpha,
    output_beta,
    output_relu
    FROM layer_params
    ORDER BY layer ASC
]]
end

local fetch_layer_data = function(db, id)
  local stmt = db:prepare [[
    SELECT
    weight,
    bias,
    half_precision
    FROM layer_data
    WHERE layer = ?
]]
  stmt:bind_values(id)
  local weight, bias
  local rc = stmt:step()
  if rc == sqlite3.ROW then
    local res = stmt:get_named_values()
    weight, bias = res.weight, res.bias
    if res.half_precision > 0 then
      weight = half_prec_to_float(weight)
      bias = half_prec_to_float(bias)
    end
  else
    print('[warn] no layer data (layer id=' .. id .. ', code=' .. rc .. ')')
  end
  stmt:finalize()
  return weight, bias
end

local fetch_mean_activity = function(db)
  local stmt = db:prepare [[
    SELECT
    mean_activity
    FROM convnet_params
    WHERE convnet = 0
  ]]
  local mean
  local rc = stmt:step()
  if rc == sqlite3.ROW then
    mean = stmt:get_value(0)
  else
    print('[warn] no mean activity (code=' .. rc .. ')')
  end
  stmt:finalize()
  return mean
end

local load_conv_weights = function(db, layer, ni, no, kw, kh)
  local weight_str, bias_str = fetch_layer_data(db, layer.id)
  assert(weight_str)
  assert(bias_str)
  local bhwd = torch.Tensor(no, kh, kw, ni)
  copy_float_str(weight_str, bhwd)
  -- deinterleave (BHWD -> BDHW)
  local weight = bhwd:permute(1, 4, 2, 3):contiguous():view(no, ni*kh*kw)
  local bias = torch.Tensor(no)
  copy_float_str(bias_str, bias)
  return weight, bias
end

local load_conv = function(db, layer, net)
  assert(layer.output_partition == 1)
  assert(layer.output_channels == layer.input_matrix_channels)
  local ni = layer.input_matrix_channels
  local no = layer.output_count
  local kw = layer.output_rows
  local kh = layer.output_cols
  local dw = layer.output_strides
  local dh = dw
  local zp = layer.output_border
  local sc = nn.SpatialConvolutionMM(ni, no, kw, kh, dw, dh, zp)
  sc.weight, sc.bias = load_conv_weights(db, layer, ni, no, kw, kh)
  net:add(sc)
  net:add(nn.ReLU())
  return ni, no, kw, kh, dw, dh, zp
end

local load_pool = function(db, layer, net)
  local kw = layer.output_size
  local kh = kw
  local dw = layer.output_strides
  local dh = dw
  local zp = layer.output_border
  if zp > 1 then
    net:add(nn.SpatialZeroPadding(zp))
  end
  if CCV_TYPES[layer.type] == 'CCV_CONVNET_MAX_POOL' then
    net:add(nn.SpatialMaxPooling(kw, kh, dw, dh))
  else
    net:add(nn.SpatialAveragePooling(kw, kh, dw, dh))
  end
  return kw, dw, zp
end

local load_fc = function(db, layer, net, fc_num, spatial)
  local ni = layer.input_node_count
  local no = layer.output_count
  if spatial then
    local kw = 1
    local kh = 1
    if fc_num == 1 then
      kw = layer.input_matrix_cols
      kh = layer.input_matrix_rows
      ni = ni/(kw*kh)
    end
    local sc = nn.SpatialConvolutionMM(ni, no, kw, kh)
    sc.weight, sc.bias = load_conv_weights(db, layer, ni, no, kw, kh)
    net:add(sc)
  else
    local li = nn.Linear(ni, no)
    local weight, bias = fetch_layer_data(db, layer.id)
    assert(weight)
    copy_float_str(weight, li.weight)
    assert(bias)
    copy_float_str(bias, li.bias)
    if fc_num == 1 then
      -- re-interleave (DHW -> HWD) to fit with ccv fully-connected data layout
      net:add(nn.Transpose({1, 2}, {2, 3}))
      net:add(nn.Reshape(ni))
    end
    net:add(li)
  end
  local relu
  if layer.output_relu > 0 then
    relu = true
    net:add(nn.ReLU())
  else
    relu = false
  end
  return ni, no, relu
end

local load_mean = function(db, input)
  local data = fetch_mean_activity(db)
  local mean, width, height
  if data then
    width, height = fetch_convnet_input_size(db)
    assert((width > 0) and (height > 0))
    assert(#data/ffi.sizeof('float') == input.channels*width*height)
    mean = torch.Tensor(height, width, input.channels)
    copy_float_str(data, mean)
    -- deinterleave (HWD -> DHW)
    mean = mean:permute(3, 1, 2):contiguous()
  end
  return mean, width, height
end

local load = function(filename, options)
  local db      = assert(sqlite3.open(filename))
  local options = options or {}
  local net     = nn.Sequential()
  local fc_num  = 1
  local log     = options.verbose and print or noop
  local input, num_output

  for layer in fetch_all_layers(db) do
    log("(" .. layer.id .. ") " .. CCV_TYPES[layer.type])
    local ch = layer.input_matrix_channels
    local iw = layer.input_matrix_cols
    local ih = layer.input_matrix_rows
    if not input then
      input = {
        channels = ch,
        width = iw,
        height = ih,
      }
      log("   input channels: " .. ch)
    end
    log("   input size: " .. iw .. "x" .. ih)
    assert(layer.input_matrix_partition == 1)
    if CCV_TYPES[layer.type] == 'CCV_CONVNET_CONVOLUTIONAL' then
      local ni, no, kw, kh, dw, dh, zp = load_conv(db, layer, net)
      log("   nb. input planes: " .. ni)
      log("   nb. output planes: " .. no)
      log("   kw: " .. kw .. ", kh: " .. kh)
      log("   stride: " .. dw .. ", pad: " .. zp)
    elseif (
      (CCV_TYPES[layer.type] == 'CCV_CONVNET_MAX_POOL') or
      (CCV_TYPES[layer.type] == 'CCV_CONVNET_AVERAGE_POOL')
    ) then
      local kw, dw, zp = load_pool(db, layer, net)
      log("   pooling size: " .. kw)
      log("   stride: " .. dw .. ", pad: " .. zp)
    elseif CCV_TYPES[layer.type] == 'CCV_CONVNET_FULL_CONNECT' then
      local ni, no, relu = load_fc(db, layer, net, fc_num, options.spatial)
      num_output = no
      fc_num = fc_num + 1
      log("   nb. input: " .. ni)
      log("   nb. output: " .. no)
      log("   relu: " .. (relu and 'yes' or 'no'))
    else
      -- TODO(cedric) handle LRN (= CCV_CONVNET_LOCAL_RESPONSE_NORM)
      error('not supported or unknown ccv module (type=' .. layer.type .. ')')
    end
    collectgarbage()
  end

  local mean = load_mean(db, input)

  db:close()

  local meta = {
    input = {
      channels = input.channels,
      width = input.width,
      height = input.height,
    },
    mean = mean,
    num_output = num_output,
  }

  return net, meta
end

return {
  load = load,
}
