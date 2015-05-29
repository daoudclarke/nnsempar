require 'json'
require 'torch'
require 'nn'

local TOKENIZE_REGEX = "[a-zA-Z]+"
local STOPWORDS = {the = true, is = true, of = true}


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


function vectorize(features, feature_indices)
   -- Return a sparse vector containing indexes of features
   local sparse = {}
   for _, feature in pairs(features) do
      local i = feature_indices[feature]
      table.insert(sparse, {i, 1.0})
   end
   return torch.Tensor(sparse)
end


function get_dataset(file_path)
   local dataset = {}
   dataset.file_path = file_path
   dataset.positions = {}
   dataset.feature_indices = {}
   dataset._size = 0

   local data_file = torch.DiskFile(file_path)
   data_file:quiet()
   local feature_index = 1
   repeat
      table.insert(dataset.positions, data_file:position())
      local line = data_file:readString('*l')
      if line == "" then break end
      dataset._size = dataset._size + 1
      local data = json.decode(line)
      local features = get_features(data)
      for i, feature in pairs(features) do
	 if dataset.feature_indices[feature] == nil then
	    print(feature, feature_index)
	    dataset.feature_indices[feature] = feature_index
	    feature_index = feature_index + 1
	 end	 
      end
      dataset.num_features = feature_index - 1
   until false

   function get_vector(t, k)
      local position = t.positions[k]
      local file = torch.DiskFile(file_path)
      file:seek(position)
      local line = file:readString('*l')
      local data = json.decode(line)
      local features = get_features(data)
      return vectorize(features, t.feature_indices)
   end

   dataset.size = function()
      return dataset._size
   end
   
   dataset = setmetatable(dataset, {__index = get_vector})
   return dataset
end


function get_model(num_features)
   -- Logistic regression network
   -- A sparse input
   -- A softmax layer

   local model = nn.Sequential()
   model:add( nn.SparseLinear(num_features, 2) )
   model:add( nn.LogSoftMax() )
   
   return model
end


-- local data_file_path = '../fbsearch/working/prepared-head.json'
-- local feature_indices, num_features = get_feature_indices(data_file_path)
-- print("Number of features found: ", num_features)

local data_file_path = '../fbsearch/working/prepared-head.json'
local dataset = get_dataset(data_file_path)
print("Number of features found: ", dataset.num_features)
print("First element: ", dataset[1])
print("Last element: ", dataset[dataset:size()])

local model = get_model(dataset.num_features)
local criterion = nn.ClassNLLCriterion()

