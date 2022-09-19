local inspect = require("inspect")
local fmt = string.format
require("tnn")
require("util")

local fn = function(a, b)
   if a == 1 and b == 0 then return 1 end
   if a == 0 and b == 1 then return 1 end
   return 0
end
local net_opts = {
   neuron_counts = {2, 3, 1},
   act_fns = {sigmoid, sigmoid},
   d_act_fns = {d_sigmoid, d_sigmoid}
}
local train_opts = {
   shuffle = false,
   epochs = 1,
   learning_rate = 0.1,
   log_freq = 0.01
}

-- Initialize neural net
print("Net")
math.randomseed(6929)
local net = new_net(net_opts)
print(inspect(net))

-- Generate training data
local training_data = {
   { inputs={1,1}, outputs={0} },
   { inputs={1,0}, outputs={1} },
   { inputs={0,1}, outputs={1} },
   { inputs={0,0}, outputs={0} }
}
local testing_data = training_data

-- Train
print("Training")
train(net, training_data, train_opts)

-- Test
print("Testing")
for i, _ in ipairs(testing_data) do
   print(fmt("test %i: %s", i, inspect(testing_data[i])))
   local pred = ff(net, testing_data[i].inputs)
   print(fmt("prediction %i: %s", i, inspect(pred)))
end
