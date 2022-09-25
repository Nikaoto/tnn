local inspect = require("inspect")
require("util")

-- for debugging
local k=false
local function printk(msg)
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
   net.i = tbl_alloc(neuron_counts[1], 1)
   net.h = tbl_alloc(neuron_counts[2], 2)
   net.o = tbl_alloc(neuron_counts[3], 3)

   -- input -> hidden weights
   net.ihw = rand_arr(wmin, wmax, #net.i * #net.h)

   -- hidden -> output weights
   net.how = rand_arr(wmin, wmax, #net.h * #net.o)

   -- biases (only in the hidden layer)
   net.b = rand_arr(wmin, wmax, neuron_counts[2])

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
      sum = sum + net.b[i]
      net.h[i] = net.act_fns[1] and net.act_fns[1](sum) or sum
   end

   -- feed from hidden to output layer
   for i, _ in ipairs(net.o) do
      local sum = 0
      for j, _ in ipairs(net.h) do
         sum = sum + net.h[j] * net.how[i + (j-1) * #net.o]
      end
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
         local bp_nudges = bp(net, data, learning_rate)
         --local grad_nudges = grad(net, data, learning_rate, step)

         -- compare bp and grad nudges
         -- local diff = cmp_arr(bp_nudges, grad_nudges, function(a, b)
         --     if a == nil and b == nil then return true end
         --     if a == nil or b == nil then return false end
         --     return (math.abs(a - b) < 0.0000001)
         -- end)
         printk("bp() and grad() nudge differences:")
         printk(inspect(diff))
      end
   
      -- log status
      if iter % log_every == 0 then
         print(fmt("epoch = %i, avg_loss = %f", iter, avg_loss))
      end
   end
end

-- numerically calculates the partial derivatives for each weight with respect
-- to the error. uses ff() many times and is therefore slow(er than bp()) but
-- works just as well while being much simpler.
-- returns array of nudges (for debugging purposes)
function grad(net, data, rate, diff_step)
   local nudges = {}
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

      how_gvec[i] = grad
      table.insert(nudges, -grad)
      printk(fmt("grad(): net.how[%d] nudge is %g", i, -grad))
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
      table.insert(nudges, -grad)
      printk(fmt("grad(): net.ihw[%d] nudge is %g", i, -grad))
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
      table.insert(nudges, -grad)
   end

   -- tune weights
   for i, _ in ipairs(how_gvec) do
      net.how[i] = net.how[i] - how_gvec[i]
   end
   for i, _ in ipairs(ihw_gvec) do
      net.ihw[i] = net.ihw[i] - ihw_gvec[i]
   end

   -- tune biases
   for i, _ in ipairs(b_gvec) do
      net.b[i] = net.b[i] - b_gvec[i]
   end

   return nudges
end

-- uses backprop with derivatives, much faster than grad().
-- returns array of nudges (for debugging purposes)
function bp(net, data, rate)
   local nudges = {}
   local bias_nudges = {}
   local out = data.outputs

   -- tune hidden -> output weights
   local sum = {}
   for wi, _ in ipairs(net.how) do
      local oni = 1 + (wi-1) % #net.o
      local hni = 1 + math.floor((wi-1) / #net.o)

      -- (pd error) / (pd output)
      local e = d_err(net.o[oni], out[oni])

      -- (pd output) / (pd activation)
      local a = net.d_act_fns[2] and net.d_act_fns[2](net.o[oni])
                                  or 1

      -- (pd activation) / (pd neuron)
      sum[hni] = (sum[hni] or 0) + e * a * net.how[wi]

      -- (pd activation) / (pd weight) == net.h[hni]
      local grad = e * a * net.h[hni]

      printk(("ni1=%i, ni2=%i, li=%i, wi=%i, e=%g, a=%g, grad=%g")
               :format(hni, oni, 2, wi, e, a, grad * -rate))

      printk(fmt("bp(): net.how[%d] nudge is %g", wi, -grad * rate))
      table.insert(nudges, -grad * rate)
      net.how[wi] = net.how[wi] - grad * rate
   end

   -- tune input -> hidden weights
   for wi, _ in ipairs(net.ihw) do
      local hni = 1 + (wi-1) % #net.h
      local ini = 1 + math.floor((wi-1) / #net.h)
      local a = net.d_act_fns[1] and net.d_act_fns[1](net.h[hni])
                                 or  1
      local grad = a * sum[hni] * net.i[ini]
      printk(fmt("bp(): net.ihw[%d] nudge is %g", wi, -grad * rate))
      table.insert(nudges, -grad * rate)
      net.ihw[wi] = net.ihw[wi] - grad * rate

      -- tune biases in hidden layer
      local bn = -a * sum[hni] * rate
      table.insert(bias_nudges, bn)
      net.b[hni] = net.b[hni] + bn
   end

   for _, v in ipairs(bias_nudges) do
      table.insert(nudges, v)
   end

   return nudges
end
