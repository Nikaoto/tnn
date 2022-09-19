-- util

function cmp_arr(arr1, arr2, eq_fn)
   local diff = {}
   for k, v in pairs(arr1) do
      if eq_fn(arr2[k], v) == false then
         table.insert(diff, {key = k, bp = v, grad = arr2[k]})
      end
   end
   return diff
end

function shuffle(arr)
   for i=#arr, 1, -1 do
      local j = math.random(1, i)
      arr[j], arr[i] = arr[i], arr[j]
   end
end

function sq(n) return n*n end

function lerp(a, b, p)
   return a + (b - a) * p
end

function randf(min, max)
   return lerp(min, max, math.random(0, 10000) / 10000)
end

function map(arr, fn)
   local new_arr = {}
   for i, _ in ipairs(arr) do
      new_arr[i] = fn(arr[i], i)
   end
   return new_arr
end

function tbl_alloc(n, val)
   local tbl = {}
   for i=1, n do table.insert(tbl, val) end
   return tbl
end

function rand_arr(min, max, n)
   local arr = {}
   for i=1, n do
      table.insert(
         arr,
         lerp(min, max, math.random(0, 1000) / 1000))
   end
   return arr
end

function linear(w)
   return w
end

function relu(a)
   return a > 0 and a or 0
end

function sigmoid(a)
   return 1 / (1 + math.exp(-a))
end

function tanh(a)
   return math.tanh(a)
end

function d_relu(a)
   return a > 0 and 1 or 0
end

-- 'a' here is already the sigmoid of the neuron activation
function d_sigmoid(a)
   return a * (1 - a)
end
