local inspect = require("inspect")
require("util")

k=true
function printk(msg)
   if k then print(msg) end
end

-- Defaults
local def = {
   act_fns = {},
   d_act_fns = {},
   diff_step = 0.0001,
   learning_rate = 0.000001,
   epochs = 10000,
   shuffle_training_data = false,
   training_log_freq = 0.01,
   wmin = -5,
   wmax = 5
}

local fmt = string.format

function err(actual, desired)
   return sq(actual - desired)
end

function d_err(actual, desired)
   return 2 * (actual - desired)
end

function total_err(actual, desired)
   local loss = 0
   for i, _ in ipairs(actual) do
      loss = loss + err(actual[i], desired[i])
   end
   return loss
end

function new_net(opts)
   local neuron_counts = opts.neuron_counts
   local wmin          = opts.wmin or def.wmin
   local wmax          = opts.wmax or def.wmax
   local act_fns       = opts.act_fns    or def.act_fns
   local d_act_fns     = opts.d_act_fns  or def.d_act_fns

   local net = {
      act_fns = act_fns,
      d_act_fns = d_act_fns,
   }

   -- neuron layers
   net.i = tbl_alloc(neuron_counts[1], 0)
   net.h = tbl_alloc(neuron_counts[2], 1)
   net.o = tbl_alloc(neuron_counts[3], 2)

   -- input -> hidden weights
   net.ihw = rand_arr(wmin, wmax, #net.i * #net.h)

   -- hidden -> output weights
   net.how = rand_arr(wmin, wmax, #net.h * #net.o)

   -- biases
   net.b = rand_arr(wmin, wmax, 3)

   return net
end

function ff(net, inp)
   -- insert inputs
   if inp then
      if #inp ~= #net.i then error("#inp ~= #net.i") end
      for i, _ in ipairs(inp) do
         net.i[i] = inp[i]
      end
   end

   -- feed from input to hidden layer
   for i, _ in ipairs(net.h) do
      local sum = 0
      for j, _ in ipairs(net.i) do
         sum = sum + net.i[j] * net.ihw[i + (j-1) * #net.h]
      end
      sum = sum + net.b[2]
      net.h[i] = net.act_fns[1] and net.act_fns[1](sum) or sum
   end

   -- feed from hidden to output layer
   for i, _ in ipairs(net.o) do
      local sum = 0
      for j, _ in ipairs(net.h) do
         sum = sum + net.h[j] * net.how[i + (j-1) * #net.o]
      end
      sum = sum + net.b[3]
      net.o[i] = net.act_fns[2] and net.act_fns[2](sum) or sum
   end

   -- return output layer
   return net.o
end

function train(net, training_data, opts)
   if not opts then opts = {} end
   local learning_rate = opts.learning_rate or def.learning_rate
   local step          = opts.diff_step or def.diff_step
   local epochs        = opts.epochs or def.epochs
   local shuf          = opts.shuffle or def.shuffle_training_data
   local log_freq      = opts.log_freq or def.training_log_freq
   local log_every     = 1 / log_freq

   if shuf then shuffle(training_data) end

   for iter=1, opts.epochs do
      local avg_loss = 0
      for _, data in ipairs(training_data) do
         local out = ff(net, data.inputs)

         -- update average loss
         local loss = total_err(out, data.outputs)
         avg_loss = avg_loss + loss / #training_data

         -- backprop
         bp(net, data, learning_rate)
         grad(net, data, learning_rate, step)
      end
   
      -- log status
      if iter % log_every == 0 then
         print(fmt("epoch = %i, avg_loss = %f", iter, avg_loss))
      end
   end
end

-- uses ff many times, slower
function grad(net, data, rate, diff_step)
   local how_gvec = {}
   local ihw_gvec = {}
   local b_gvec = {}

   for i, _ in ipairs(net.how) do
      local step = diff_step
      local loss1 = total_err(ff(net, data.inputs), data.outputs)
      local w1 = net.how[i]
      local w2 = net.how[i] + step
      net.how[i] = w2
      local loss2 = total_err(ff(net, data.inputs), data.outputs)
      local grad = rate * (loss2 - loss1) / step
      net.how[i] = w1

      printk(fmt("grad(): net.how[%d] nudge is %f", i, -grad))
      how_gvec[i] = grad
   end

   for i, _ in ipairs(net.ihw) do
      local step = diff_step
      local loss1 = total_err(ff(net, data.inputs), data.outputs)
      local w1 = net.ihw[i]
      local w2 = net.ihw[i] + step
      net.ihw[i] = w2
      local loss2 = total_err(ff(net, data.inputs), data.outputs)
      local grad = rate * (loss2 - loss1) / step
      net.ihw[i] = w1

      ihw_gvec[i] = grad
      printk(fmt("grad(): net.ihw[%d] nudge is %f", i, -grad))
   end

   for i, _ in ipairs(net.b) do
      local step = diff_step
      local loss1 = total_err(ff(net, data.inputs), data.outputs)
      local b1 = net.b[i]
      local b2 = net.b[i] + step
      net.b[i] = b2
      local loss2 = total_err(ff(net, data.inputs), data.outputs)
      local grad = rate * (loss2 - loss1) / step
      net.b[i] = b1

      b_gvec[i] = grad   
   end

   -- -- tune weights
   -- for i, _ in ipairs(how_gvec) do
   --    net.how[i] = net.how[i] - how_gvec[i]
   -- end
   -- for i, _ in ipairs(ihw_gvec) do
   --    net.ihw[i] = net.ihw[i] - ihw_gvec[i]
   -- end

   -- -- tune biases
   -- for i, _ in ipairs(b_gvec) do
   --    net.b[i] = net.b[i] - b_gvec[i]
   -- end
end


-- TODO: compare grad() and bp() nudges:
--       - output an array of gradients for each fn
--       - compare the two arrays
--       - point out any differences

-- TODO: tune biases as well
-- uses backprop with derivatives, faster
function bp(net, data, rate)
   local out = data.outputs

   -- tune hidden -> output weights
   local sum = {}
   for wi, _ in ipairs(net.how) do
      local hni = 1 + math.floor((wi-1) / #net.o)
      local oni = 1 + (wi-1) % #net.o
      -- (pd error) / (pd output)
      local e = d_err(net.o[oni], out[oni])
      -- (pd output) / (pd activation)
      local a = net.d_act_fns[2] and net.d_act_fns[2](net.o[oni])
                                  or 1
      -- (pd activation) / (pd weight) == net.h[hni]
      local grad = e * a * net.h[hni]
      sum[hni] = (sum[hni] or 0) + grad

      printk(fmt("bp(): net.how[%d] nudge is %f", wi, -grad * rate))
      --net.how[wi] = net.how[wi] - grad * rate
   end

   -- tune input -> hidden weights
   for wi, _ in ipairs(net.ihw) do
      local hni = 1 + (wi-1) % #net.h
      local ini = 1 + math.floor((wi-1) / #net.h)
      local a = net.d_act_fns[1] and net.d_act_fns[1](net.h[hni])
                                 or  1
      local grad = a * sum[hni] * net.i[ini]
      printk(fmt("bp(): net.ihw[%d] nudge is %f", wi, -grad))
      --net.ihw[wi] = net.ihw[wi] - grad * rate
   end
end