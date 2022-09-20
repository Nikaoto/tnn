local inspect = require("inspect")
local fmt = string.format
require("tnn")
require("util")

-- this net will discover the XOR function
local net_opts = {
   neuron_counts = {2, 3, 1},
   act_fns = {sigmoid, sigmoid},
   d_act_fns = {d_sigmoid, d_sigmoid}
}
local train_opts = {
   shuffle = false,
   epochs = 2500,
   learning_rate = 0.1,
   log_freq = 0.01
}

-- initialize neural net
print("\nNet")
math.randomseed(6929)
local net = new_net(net_opts)
print(inspect(net))

-- training data (XOR function)
local training_data = {
   { inputs={1,1}, outputs={0} },
   { inputs={1,0}, outputs={1} },
   { inputs={0,1}, outputs={1} },
   { inputs={0,0}, outputs={0} }
}
local testing_data = training_data

-- train
print("\nTraining")
train(net, training_data, train_opts)

-- test
print("\nTesting")
for i, _ in ipairs(testing_data) do
   print(fmt("test %i: %s", i, inspect(testing_data[i])))
   local pred = ff(net, testing_data[i].inputs)
   print(fmt("prediction %i: %s", i, inspect(pred)))
end
