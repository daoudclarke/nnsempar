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
      if i ~= nil then table.insert(sparse, {i, 1.0}) end
   end
   return torch.Tensor(sparse)
end


function get_data_line(data_file, position)
   data_file:seek(position)
   local line = data_file:readString('*l')
   if line == "" then return end
   return JSON:decode(line)
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

   dataset.next_test_position = data_file:position()
   -- dataset.next_test_position = 1

   function get_example(t, k)
      local position = t.positions[k]
      -- local file = torch.DiskFile(file_path)
      local data = get_data_line(data_file, position)
      local features = get_features(data)
      local vector = vectorize(features, t.feature_indices)
      local label = 1
      if data.score > 0.0 then label = 2 end
      return {vector, label}
   end

   function dataset:size()
      return self._size
   end

   function dataset:next_test_items()
      local data = get_data_line(data_file, self.next_test_position)
      if data == nil then return end
      local current_source = data.source
      local items = {}
      while current_source == data.source do
	 table.insert(items, data)
	 data = get_data_line(data_file, data_file:position())
	 if data == nil then return end
      end
      self.next_test_position = data_file:position()
      return items
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


function test(dataset, model, num_to_test)
   chosen_scores = {}
   for i=1, num_to_test do
      items = dataset:next_test_items()
      if items == nil then break end
      local best_score = nil
      local best_item = {score = 0.0}
      for i, data in pairs(items) do
	 local features = get_features(data)
	 local vector = vectorize(features, dataset.feature_indices)
	 if vector:dim() ~= 0 then
	    local score = model:forward(vector)[1]
	    -- print(data.score, data.value[1], score)
	    if best_score == nil or score < best_score then
	       best_score = score
	       best_item = data
	    end
	 end
      end
      print("Best item: ", best_item.value[1], best_score)
      table.insert(chosen_scores, best_item.score)
   end
   local num_results = #chosen_scores
   chosen_scores = torch.Tensor(chosen_scores)
   print(chosen_scores:view(1, -1))
   stderr = torch.std(chosen_scores) / math.sqrt(num_results) 
   print("Num results:", num_results, "Mean:", torch.mean(chosen_scores), "+/-", stderr)
end


-- local data_file_path = '../fbsearch/working/prepared-head.json'
-- local feature_indices, num_features = get_feature_indices(data_file_path)
-- print("Number of features found: ", num_features)

local data_file_path = '../fbsearch/working/prepared.json'
local num_train_examples = 70
local dataset = get_dataset(data_file_path, num_train_examples)
print("Number of features found: ", dataset.num_features)
-- print("First element: ", dataset[1])
-- print("Last element: ", dataset[dataset:size()])

local model = get_model(dataset.num_features)
local criterion = nn.ClassNLLCriterion(torch.Tensor({0.001, 0.999}))


local trainer = nn.StochasticGradient(model, criterion)
trainer.learningRate = 0.01
trainer.maxIteration = 30
trainer:train(dataset)
for i, v in pairs(model) do
   print(i, v)
end

test(dataset, model, 100)

-- model:forward(dataset[1])

