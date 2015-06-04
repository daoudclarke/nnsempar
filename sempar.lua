-- require 'json'
require 'torch'
require 'nn'

JSON = assert(loadfile "JSON.lua")()

local TOKENIZE_REGEX = "[a-zA-Z]+"
local STOPWORDS = {the = true, is = true, of = true, ['in'] = true}


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


function get_dataset(file_path, num_train_examples)
   local dataset = {}
   dataset.file_path = file_path
   dataset.positions = {}
   dataset.feature_indices = {}
   dataset._size = 0

   local data_file = torch.DiskFile(file_path)
   data_file:quiet()
   local feature_index = 1
   local num_seen_examples = 0
   local last_seen_source = ''
   repeat
      table.insert(dataset.positions, data_file:position())
      local line = data_file:readString('*l')
      if line == "" then break end
      local data = JSON:decode(line)
      if data.source ~= last_seen_source then
	 num_seen_examples = num_seen_examples + 1
	 if num_seen_examples > num_train_examples then
	    break
	 end
	 last_seen_source = data.source
      end

      dataset._size = dataset._size + 1
      local features = get_features(data)
      for i, feature in pairs(features) do
	 if dataset.feature_indices[feature] == nil then
	    dataset.feature_indices[feature] = feature_index
	    feature_index = feature_index + 1
	    if (feature_index % 1000 == 0) then
	       print(feature, feature_index)
	    end
	 end	 
      end
      dataset.num_features = feature_index - 1
   until false

   function get_example(t, k)
      local position = t.positions[k]
      -- local file = torch.DiskFile(file_path)
      data_file:seek(position)
      local line = data_file:readString('*l')
      local data = JSON:decode(line)
      local features = get_features(data)
      local vector = vectorize(features, t.feature_indices)
      local label = 1
      if data.score > 0.0 then label = 2 end
      return {vector, label}
   end

   dataset.size = function()
      return dataset._size
   end
   
   dataset = setmetatable(dataset, {__index = get_example})
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

local data_file_path = '../fbsearch/working/prepared.json'
local num_train_examples = 10
local dataset = get_dataset(data_file_path, num_train_examples)
print("Number of features found: ", dataset.num_features)
-- print("First element: ", dataset[1])
-- print("Last element: ", dataset[dataset:size()])

local model = get_model(dataset.num_features)
local criterion = nn.ClassNLLCriterion()


local trainer = nn.StochasticGradient(model, criterion)
trainer.learningRate = 0.01
trainer:train(dataset)

-- model:forward(dataset[1])

