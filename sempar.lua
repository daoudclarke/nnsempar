require 'json'
require 'torch'
require 'nn'

local TOKENIZE_REGEX = "[a-zA-Z]+"
local STOPWORDS = {the = true, is = true, of = true}

function get_data(file_path)
   function inner()
      local data_file = torch.DiskFile(file_path)
      data_file:quiet()

      repeat
	 local line = data_file:readString('*l')
	 if line == "" then break end
	 local data = json.decode(line)
	 coroutine.yield(data)
      until false
   end
   
   return coroutine.wrap(inner)
end


function get_features(example)
   -- Return an array containing feature strings
   local result = {}
   for source_token in string.gmatch(example.source, TOKENIZE_REGEX) do
      if not STOPWORDS[source_token] then
	 for target_token in string.gmatch(example.target, TOKENIZE_REGEX) do
	    if not STOPWORDS[target_token] then
	       table.insert(result, source_token .. '_' .. target_token)
	    end
	 end
      end
   end
   return result
end


function get_feature_indices(file_path)
   data_iterator = get_data(file_path)
   local feature_index = 1
   local feature_indices = {}

   for example in data_iterator do
      local features = get_features(example)
      for i, feature in pairs(features) do
	 if feature_indices[feature] == nil then
	    print(feature, feature_index)
	    feature_indices[feature] = feature_index
	    feature_index = feature_index + 1
	 end	 
      end
   end
   return feature_indices, feature_index - 1
end


function vectorize(features, feature_indices)
   -- Return a sparse vector containing indexes of features
   local sparse = {}
   for _, feature in pairs(features) do
      local i = string_indices[feature]
      sparse[i] = 1.0
   end
   return torch.Tensor(sparse)
end

-- function get_data_tensor(file_path, string_indices, max_index)
--    data_iterator = get_data(file_path)
   
--    for data in data_iterator do

--    end
-- end


-- function get_network(vocab_size)
--    -- Logistic regression network
--    -- Two sparse inputs concatenated for the source and target sentences
--    -- A softmax layer

--    local model = nn.Sequential()
--    local input_layer = nn.Concat(1)
--    input_layer:add( nn.SparseLinear(vocab_size, 2) )
--    input_layer:add( nn.SparseLinear(vocab_size, 2) )
--    model:add(input_layer)
   

-- end


local data_file_path = '../fbsearch/working/prepared-head.json'
local feature_indices, num_features = get_feature_indices(data_file_path)
print("Number of features found: ", num_features)
