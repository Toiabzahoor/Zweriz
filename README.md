# Zweriz (v0.3.0)

A fast, dynamic programming language with native, seamless GPU acceleration—built for high-performance computing, Deep Learning, and now with structured classes and robust error handling.

Zweriz is a modern scripting language designed to make heavy computing as easy as writing a simple script. Whether you are building a quick math tool, simulating millions of particles, or training a neural network, Zweriz writes like a standard dynamic language but gives you the power to offload heavy math directly to your NVIDIA GPU with a single block of code. 

## A Note from the Creator

Hi! I'm the solo teenage developer behind Zweriz. Welcome to the v0.3.0 release! A massive thank you to everyone who tested the earlier versions and reported bugs. 

In this version, Zweriz matures significantly. I've introduced **Classes** (which act as clean factory containers for dictionaries), robust **Try/Catch error handling**, memory-mapped files via `mmap` for instant loading of massive datasets, and new native string/array extensions. You can now also write multiple statements on a single line using semicolons, making scripts cleaner and more compact.

If you are interested in languages, compilers, or GPU tech, I would massively appreciate volunteer bug hunters! Try to break it, and let me know what you find so we can make v0.4 even better.

## Getting Started

Currently, Zweriz is distributed as two standalone compiled binaries:
* `zweriz`: CPU only.
* `zweriz_cuda`: Features both GPU acceleration and seamless CPU fallback.

Run a script:
```bash
./zweriz_cuda my_script.zw
```

Run the Interactive REPL:
```bash
./zweriz_cuda
```

*(Note: Use `./zweriz` instead if you are using the CPU-only binary).*

**Safe Mode (Sandboxed):**
If you are running an untrusted script, use the `--safe` flag. This disables dangerous OS commands (`os.cmd`), environment variable access, and restricts file I/O to prevent sandbox escapes.
```bash
./zweriz_cuda --safe my_script.zw
```

---

## What's New in v0.3.0

* **Classes:** Create data structures with default values. Under the hood, these are elegantly parsed as factory functions returning dictionaries, keeping the language runtime fast and dynamic.
* **Error Handling:** Protect your execution flow with native `try { ... } catch (err) { ... }` blocks and custom `throw "error"` statements.
* **Memory Mapping (`mmap`):** Load massive binary arrays of 64-bit floats instantly without parsing overhead via `mmap.load_f64(path)`.
* **Semicolons / Inline Statements:** You can now separate multiple statements or class fields on a single line using semicolons.
* **Expanded Standard Library:** Deeply expanded `array` (push, pop, clone, shape, concat) and `string` (to_lower, to_upper, contains, char_code_at) manipulation modules.
* **Bitwise & Math Expansions:** New bitwise token operators (`|`, `&`, `^`, `<<`, `>>`, `~`) and native math calls like `math.popcount` for low-level optimizations like chess bitboards.

---

## Language Syntax Guide

Zweriz uses a clean, brace-based syntax. **All numeric values in Zweriz are 64-bit floats.**

### Variables & Data Types
```python
# Numbers (Always 64-bit floats, booleans act as 1.0/0.0)
x = 42.0
flag = true

# Strings and F-Strings
name = "World"
greeting = f"Hello, {name}!"

# Arrays & Matrices (Auto-shaped)
v = [0.0, 1.0, 4.0, 9.0]
X = [
    [0.0, 0.0],
    [1.0, 0.0],
    [1.0, 1.0]
]

# Dictionaries
config = { "speed": 100.0, "active": 1.0 }
```

### Classes
Classes in Zweriz act as clean data containers. They instantiate with their default field values. Functions operate on these state objects directly.

```python
class BoardState {
    grid = zeros(64)
    turn = 0.0
    castle_WK = 1.0; castle_WQ = 1.0   # Inline fields with semicolons!
}

board = BoardState()
board.turn = 1.0
```

### Control Flow & Error Handling
```python
# If / Else
if score > 90.0 {
    print("A")
} else {
    print("B")
}

# Standard & For-In Loops
for i = 0 to 10 { print(i) }
for item in v { total = total + item }

# Try / Catch
try {
    data = io.read("config.json")
} catch (err) {
    print(f"Error loading config: {err}")
    throw "Fatal configuration error!"
}
```

---

## The Magic: GPU { ... } Blocks

The defining feature of Zweriz is the `GPU {}` block. You wrap your heavy math in a GPU block, and the Zweriz engine automatically accelerates it. If a machine lacks a compatible NVIDIA GPU, Zweriz falls back to the CPU automatically.

### GPU Example: Fast Conditionals with `blend()`
```python
size = 5000000
X = (random.uniform(size) * 4.0) - 2.0
Y = (random.uniform(size) * 4.0) - 2.0
active = ones(size)

GPU {
    for step = 0 to 30 { 
        x_new = (X * X) - (Y * Y) - 0.7
        y_new = (2.0 * X * Y) + 0.27
        
        # 'blend' acts as a fast vectorized if/else: blend(condition, true, false)
        X = blend(active == 1.0, x_new, X)
        Y = blend(active == 1.0, y_new, Y)
        
        escaped = ((X * X) + (Y * Y)) > 4.0
        active = blend(escaped == 1.0, 0.0, active)
    }
}

survivors = sum(active) # sum() auto-escapes back to CPU!
print(f"Particles survived: {survivors}")
```

---

## Deep Learning & Matrices

Zweriz handles native matrix multiplication (`@`) and built-in autograd directly through the `nn` module.

```python
# Initialize Datasets and Weights
X = random.uniform(2048, 512, -1.0, 1.0)
Y = random.uniform(2048, 10, 0.0, 1.0)
W1 = random.uniform(512, 1024, -0.1, 0.1)

nn.track(W1)
epoch = 0

GPU {
    while epoch < 1000 {
        pred = nn.sigmoid(X @ W1)
        loss = (pred - Y) * (pred - Y)
        
        nn.backward(loss)
        nn.step(0.001)
        nn.zero_grad()
        
        epoch = epoch + 1.0
    }
}
```

---

## Standard Library Modules

Zweriz comes packed with highly optimized global modules powered by native Rust dispatches:

* **math**: `math.pow`, `math.floor`, `math.ceil`, `math.round`, `math.transpose`, `math.trapz`, `math.gradient`, `math.popcount`, `math.tzcnt`, `math.lzcnt`
* **random**: `random.float()`, `random.randint(min, max)`, `random.uniform(...)`, `random.normal(...)`
* **nn**: `nn.relu`, `nn.sigmoid`, `nn.softmax`, `nn.gelu`, `nn.dropout`, `nn.track`, `nn.backward`, `nn.step`, `nn.zero_grad`
* **net**: `net.http_get(url)`, `net.http_post(url, body)`, `net.tcp_listen(addr)`, `net.tcp_send(addr, data)`
* **string**: `string.len(s)`, `string.to_lower(s)`, `string.to_upper(s)`, `string.contains(s, sub)`, `string.parse_num(s)`, `string.char_code_at(s, idx)`, `string.from_char_code(n)`
* **array**: `array.clone(a)`, `array.shape(a)`, `array.push(a, val)`, `array.pop(a)`, `array.concat(a, b)`
* **mmap**: `mmap.load_f64(path)` (Zero-copy binary float loading)
* **io / os / time**: System tools (`os.cmd`, `os.env`, `os.exit`), file writing/reading (`io.read`, `io.write`, `io.append`, `io.delete`, `io.exists`), timestamps, and sleeping.

### Built-in Native Chess Extensions
For specialized tasks, Zweriz allows direct hooks to highly optimized C++/CUDA code. Native built-ins include:
* `gpu_beam_search(grid, depth, top_k, turn)`
* `chess_batch_generate(...)`

---

## Call for Bug Hunters!

As mentioned, Zweriz is an experimental playground. If you manage to crash the VM, find a memory leak, or discover a script that causes the GPU fallback to trigger incorrectly, please let me know!

**How to report bugs:**
1. Save the exact `.zw` script that caused the issue.
2. Note whether you ran it with or without `--safe`.
3. Submit an issue with the script and the error output.
