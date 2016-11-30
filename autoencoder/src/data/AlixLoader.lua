--
-- User: aris
--
local alixLoader, parent = torch.class('cfn.alixLoader', 'cfn.DataLoader')

function alixLoader:LoadRatings(conf)

   --no pre-process/post-processing
   function preprocess(x)  return (x-3)/2 end
   function postprocess(x) return 2*x+3 end

   -- step 3 : load ratings
   local ratesfile = io.open(conf.ratings, "r")


   -- Step 1 : Retrieve item ratings
   for line in ratesfile:lines() do

      local userIdStr, itemIdStr, ratingStr = line:match('(%d+),(%d+),(%d+)')
      local userId  = tonumber(userIdStr)
      local itemId  = tonumber(itemIdStr)
      local rating  = tonumber(ratingStr)

      local itemIndex = self:getItemIndex(itemId)
      local userIndex = self:getUserIndex(userId)

      rating = preprocess(rating)

      self:AppendOneRating(userIndex, itemIndex, rating)

   end
   ratesfile:close()

end

