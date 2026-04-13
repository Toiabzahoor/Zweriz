# Zweriz Language Cheatsheet



## Comments

```zw
# This is a comment (single-line only, hash to end of line)
```

---

## Variables & Types

All numeric values are **64-bit floats**. There is no integer type.

```zw
x = 42.0
name = "hello"
flag = true          # boolean literal
flag2 = false
nothing = 0.0        # also used as false
```

Booleans `true` and `false` exist as literals. In practice the codebase also uses `1.0` / `0.0` interchangeably for boolean logic.

---

## Strings

```zw
s = "hello world"
s2 = "escape: \n \t \\"

# f-string interpolation
print(f"Value is {x} and name is {name}")

# Concatenation with +
msg = "Row " + r + " col " + c
```

---

## Operators

### Arithmetic
| Operator | Meaning |
|----------|---------|
| `+` `-` `*` `/` | Basic arithmetic |
| `**` | Power / exponentiation |
| `%` | Modulo |
| `@` | Matrix multiply |
| `-x` | Unary negation |

### Compound Assignment
```zw
x += 1.0
x -= 1.0
x *= 2.0
x /= 2.0
x %= 3.0
x **= 2.0
```

### Comparison
| Operator | Meaning |
|----------|---------|
| `==` | Equal |
| `!=` | Not equal |
| `<`  `>` | Less / greater |
| `<=` `>=` | Less-or-equal / greater-or-equal |

### Logical
```zw
a and b
a or b
not a
a && b    # alternative to 'and'
a || b    # alternative to 'or'
```

### Bitwise
| Operator | Meaning |
|----------|---------|
| `&` | Bitwise AND |
| `\|` | Bitwise OR |
| `^` | Bitwise XOR |
| `~` | Bitwise NOT |
| `<<` | Left shift |
| `>>` | Right shift |

---

## Control Flow

### if / else
```zw
if x == 0.0 { print("zero") }

if x > 0.0 {
    print("positive")
} else {
    print("non-positive")
}

# Inline chained ifs (common pattern in source)
if abs_p == 1.0 { if dest == 0.0 { is_ep = 1.0 } }
```

### while
```zw
while epoch < 1000 {
    epoch += 1.0
}
```

### for — numeric range (exclusive end)
```zw
for i = 0 to 8 { print(i) }

# Nested
for r = 0 to 8 {
    for c = 0 to 8 {
        grid[r, c] = 0.0
    }
}
```

### for — iterator (for-each)
```zw
for val in arr {
    total = total + val
}
```

### break / continue
```zw
while true {
    if done { break }
    if skip { continue }
}
```

---

## Functions

```zw
fn add(a, b) {
    return a + b
}

fn factorial(n) {
    result = 1.0
    for i = 1 to n + 1 {
        result = result * i
    }
    return result
}

result = add(3.0, 4.0)
```

- All parameters are positional, no defaults.
- Functions can return any value. `return` is explicit.

---

## Classes

```zw
class BoardState {
    grid = zeros(64)
    turn = 0.0
    castle_WK = 1.0; castle_WQ = 1.0   # semicolons separate fields on one line
    ep_r = -1.0; ep_c = -1.0
}

# Instantiate
board = BoardState()

# Field access / mutation
board.turn = 1.0
v = board.grid[0.0]
```

- Fields are defined with default values directly in the class body.
- No constructor method — defaults run at instantiation.
- No methods — use standalone `fn` functions that accept the object as a parameter.

---

## Arrays & Tensors

### Creation
```zw
# 1D
a = zeros(64)
b = ones(8)

# 2D
m = zeros(8, 8)
m = ones(8, 8)

# Array literal — 1D
v = [0.0, 1.0, 4.0, 9.0, 16.0]

# Nested literal — auto-shaped into 2D tensor
X = [
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0]
]
```

### Indexing & Slicing
```zw
# 1D index (index is a float)
val = arr[5.0]
arr[idx] = 0.0

# 2D index
val = matrix[r, c]
matrix[7, 6] = 1.0

# Slice — returns sub-array
sub = arr[0.0 : count * 4.0]

# Length
n = len(arr)
```

### Built-in Element-wise Math (work on scalars and tensors)
```zw
sin(x)
cos(x)
exp(x)
log(x)
sqrt(x)
abs(x)
```

### Tensor Reductions (CPU-side, auto-escape from GPU blocks)
```zw
sum(tensor)          # scalar sum of all elements
mean(tensor)         # scalar mean
max(tensor, axis)    # max along axis (0 = cols, 1 = rows)
min(tensor)          # scalar min
```

### blend() — vectorised conditional
```zw
# blend(condition_tensor, value_if_true, value_if_false)
empty = blend(g2 == 0.0, ones(8,8), zeros(8,8))

# Nested blend = chained conditions
opp_B_Q = blend(g2 == -3.0, ones(8,8),
           blend(g2 == -5.0, ones(8,8), zeros(8,8)))
```

### Matrix Multiply
```zw
C = A @ B
```

---

## Dictionaries

```zw
d = { "key": value, "other": 42.0 }
```

---

## Imports

```zw
import chess
import engine

# Functions from the imported module are available directly
board = init_board()
move  = find_best_move(board)
```

---

## GPU Blocks

```zw
GPU {
    # All tensor math dispatched to VRAM
    Z1 = X @ W1
    A1 = nn.relu(Z1)
    loss = (pred - Y) * (pred - Y)

    # Reductions auto-escape back to CPU
    total = sum(loss)

    nn.backward(loss)
    nn.step(lr)
    nn.zero_grad()
}
```

- Code inside `GPU { }` runs on the GPU.
- Reductions (`sum`, `mean`, `max`, `min`) are automatically stripped out and executed on CPU.

---

## Error Handling

```zw
try {
    data = io.read("file.json")
} catch (err) {
    print(f"Error: {err}")
}

# Manual throw
throw "Data Corruption Detected!"
```

Catches: IO errors, VM type errors, bad indexing, arithmetic failures.

---

## print

```zw
print("hello")
print(f"x is {x}")
print(some_variable)
```

`print` is a keyword/statement, not a regular function.

---

## Standard Library

### `random`
| Call | Returns | Description |
|------|---------|-------------|
| `random.float()` | number | Random float in [0, 1) |
| `random.uniform(rows, cols, min, max)` | 2D array | Uniform random matrix |
| `random.uniform(size, min, max)` | 1D array | Uniform random vector |
| `random.normal(rows, cols, mean, std)` | 2D array | Normal (Gaussian) random matrix |
| `random.randint(min, max)` | number | Random integer (as float) |

### `math`
| Call | Description |
|------|-------------|
| `math.floor(x)` | Floor |
| `math.ceil(x)` | Ceiling |
| `math.round(x)` | Round to nearest |
| `math.pow(base, exp)` | Power (also `**` operator) |
| `math.min(a, b)` | Scalar min of two numbers |
| `math.max(a, b)` | Scalar max of two numbers |
| `math.transpose(matrix)` | Matrix transpose |
| `math.trapz(array, dx)` | Numerical integration (trapezoidal rule) |
| `math.gradient(array, dx)` | Numerical derivative |
| `math.shift2d(tensor, dr, dc)` | Shift 2D tensor by (dr, dc) — wraps at edges |
| `math.popcount(n)` | Count set bits |
| `math.tzcnt(n)` | Count trailing zeros (find LSB) |
| `math.lzcnt(n)` | Count leading zeros (find MSB) |

### `nn`
| Call | Description |
|------|-------------|
| `nn.track(W)` | Register weight tensor for autograd |
| `nn.backward(loss)` | Run backpropagation |
| `nn.step(lr)` | Gradient descent weight update |
| `nn.zero_grad()` | Zero all tracked gradients |
| `nn.relu(x)` | ReLU activation |
| `nn.sigmoid(x)` | Sigmoid activation |
| `nn.softmax(x)` | Softmax (numerically stable) |
| `nn.tanh(x)` | Tanh activation |
| `nn.gelu(x)` | GELU activation (tanh approximation) |
| `nn.leaky_relu(x, alpha)` | Leaky ReLU (default alpha = 0.01) |
| `nn.swish(x)` / `nn.silu(x)` | Swish / SiLU activation |
| `nn.softplus(x)` | Softplus (smooth ReLU) |
| `nn.dropout(x, p)` | Inverted dropout (default p = 0.5) |

### `string`
| Call | Description |
|------|-------------|
| `string.len(s)` | Byte length of string |
| `string.to_lower(s)` | Lowercase |
| `string.to_upper(s)` | Uppercase |
| `string.contains(s, sub)` | Returns 1.0 if s contains sub |
| `string.parse_num(s)` | Parse string to float |
| `string.char_code_at(s, i)` | UTF-32 code of character at index i |
| `string.from_char_code(n)` | Character from UTF-32 code |

### `array`
| Call | Description |
|------|-------------|
| `array.clone(a)` | Deep copy |
| `array.shape(a)` | Returns `[rows, cols]` array |
| `array.push(a, val)` | Returns new array with val appended |
| `array.pop(a)` | Returns new array with last element removed |
| `array.concat(a, b)` | Returns concatenation of two arrays |

### `io`
| Call | Description |
|------|-------------|
| `io.read(path)` | Read file to string |
| `io.read()` | Read line from stdin |
| `io.write(path, content)` | Write string to file |
| `io.append(path, content)` | Append string to file |
| `io.exists(path)` | Returns 1.0 if file exists |
| `io.delete(path)` | Delete a file |

### `os`
| Call | Description |
|------|-------------|
| `os.cmd(command)` | Run shell command, returns stdout |
| `os.env(var)` | Get environment variable |
| `os.exit(code)` | Exit with code |

### `net`
| Call | Description |
|------|-------------|
| `net.http_get(url)` | HTTP GET, returns body string |
| `net.http_post(url, body)` | HTTP POST, returns body string |
| `net.tcp_send(addr, data)` | Send string over raw TCP |
| `net.tcp_listen(addr)` | Block, accept one TCP connection, return data |

### `time`
| Call | Description |
|------|-------------|
| `time()` | Unix timestamp as float (opcode-level) |
| `time.now()` | Unix timestamp as float |
| `time.sleep(seconds)` | Sleep |

### `mmap`
| Call | Description |
|------|-------------|
| `mmap.load_f64(path)` | Memory-map a binary file of f64 values into a 1D array |

### Chess / GPU (built-in special)
| Call | Description |
|------|-------------|
| `gpu_beam_search(grid, depth, top_k, turn)` | Native GPU beam search, returns 4-element move array |
| `chess_batch_generate(...)` | Batched chess move generation on GPU |

### Misc
```zw
gc()    # Run garbage collector
```

---

## Multiple Statements Per Line

Semicolons separate statements or class field declarations on one line:

```zw
a = 1.0; b = 2.0; c = 3.0
castle_WK = 1.0; castle_WQ = 1.0
```

---

## Common Patterns

### Manual backprop (no autograd)
```zw
for i = 0 to epochs {
    Z1 = X @ W1;  A1 = nn.sigmoid(Z1)
    Z2 = A1 @ W2; A2 = nn.sigmoid(Z2)
    dZ2 = A2 - Y
    dW2 = math.transpose(A1) @ dZ2
    dA1 = dZ2 @ math.transpose(W2)
    dZ1 = dA1 * A1 * (1.0 - A1)
    dW1 = math.transpose(X) @ dZ1
    W1 = W1 - (dW1 * lr)
    W2 = W2 - (dW2 * lr)
}
```

### Autograd training loop
```zw
nn.track(W1); nn.track(W2)
epoch = 0
GPU {
    while epoch < epochs {
        pred = nn.sigmoid(X @ W1 @ W2)
        loss = (pred - Y) * (pred - Y)
        nn.backward(loss)
        nn.step(lr)
        nn.zero_grad()
        epoch = epoch + 1
    }
}
```

### Vectorised 2D mask (chess / image ops)
```zw
g2 = zeros(8, 8)
for r = 0 to 8 { for c = 0 to 8 { g2[r, c] = board.grid[r*8.0 + c] } }

white_pawns = blend(g2 == 1.0, ones(8,8), zeros(8,8))
shifted     = math.shift2d(white_pawns, -1.0, 0.0)   # move up one row
```

### Timing
```zw
t = time()
# ... work ...
print(time() - t)

# or
start = time.now()
print(time.now() - start)
```
