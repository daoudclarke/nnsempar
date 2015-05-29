require 'json'


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


function get_term_indices(file_path)
   data_iterator = get_data(file_path)
   local string_index = 1
   local string_indices = {}

   for data in data_iterator do
      local sentences = {data['source'], data['target']}
      for i, sentence in pairs(sentences) do
	 for s in string.gmatch(sentence, "[a-zA-Z]+") do
	    if string_indices[s] == nil then
	       print(s, string_index)
	       string_indices[s] = string_index
	       string_index = string_index + 1
	    end
	 end
      end
   end
end


local data_file_path = '../fbsearch/working/prepared-head.json'
get_term_indices(data_file_path)
