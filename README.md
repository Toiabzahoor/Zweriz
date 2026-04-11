# Zweriz (v0.2.0)
A fast, dynamic programming language with native, seamless GPU acceleration—now with built-in Deep Learning and Networking.

Zweriz is a modern scripting language designed to make high-performance computing as easy as writing a simple script. Whether you are building a quick math tool, simulating millions of particles, or training a neural network, Zweriz writes like a standard dynamic language but gives you the power to offload heavy math directly to your NVIDIA GPU with a single block of code.

## A Note from the Creator
Hi! I'm the solo teenage developer behind Zweriz. Welcome to the v0.2.0 release! A massive thank you to everyone who tested the initial v0.1 release and reported bugs. 

In this version, I have massively expanded Zweriz's capabilities by adding native Neural Network support, a networking module, and a brand new VS Code extension for syntax highlighting. Because I am building this on my own and don't have the budget to hire a QA team, there are bound to be undiscovered edge cases. I am releasing the compiled binary so people can play with it, test it, and push it to its limits.

**Note:** The Interactive REPL is currently in its early stages and has some known syntax parsing quirks compared to running a `.zw` file directly. 

If you are interested in languages, compilers, or GPU tech, I would massively appreciate volunteer bug hunters! Try to break it, and let me know what you find so we can make v0.3 even better.

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

### New in v0.2: VS Code Extension
Zweriz now has official editor support! You can install the Zweriz Language Support extension (`zweriz.vsix`) to get full syntax highlighting and snippets for `.zw` files. 

## Language Syntax Guide
Zweriz uses a clean, brace-based syntax that feels familiar if you have used languages like JavaScript, Python, or Rust.

### Variables & Data Types
Zweriz is dynamically typed.

```python
# Numbers (Everything is a high-precision float)
x = 42
y = 3.1415

# Strings and F-Strings
name = "World"
greeting = f"Hello, {name}!"

# Arrays & Matrices
my_array = [1, 2, 3, 4]
matrix = [[1, 2], [3, 4]]

# Dictionaries
config = {
    "speed": 100,
    "active": true
}
```

### Control Flow
```python
# If / Else
if score > 90 {
    print("A")
} else {
    print("B")
}

# Standard For Loops
for i = 0 to 10 {
    print(i)
}

# For-In Loops (Iterating Arrays)
for item in my_array {
    print(item)
}

# While Loops
while power > 0 {
    power = power - 1
}
```

## The Magic: GPU { ... } Blocks
The defining feature of Zweriz is the `GPU {}` block. You do not need to know CUDA, C++, or memory management to use your graphics card. You just wrap your heavy math in a GPU block, and the Zweriz engine automatically accelerates it.

If a machine does not have a compatible NVIDIA GPU, Zweriz implements Seamless CPU Fallback—your code will still run perfectly on the CPU without you needing to change a single line of code.

### GPU Example: Raymarching / Particle Physics
```python
# 1. Set up massive arrays on the CPU
size = 5000000
X = (random.uniform(size) * 4.0) - 2.0
Y = (random.uniform(size) * 4.0) - 2.0
active = ones(size)
escape_limit = 4.0

# 2. Wrap heavy math in the GPU block
GPU {
    for step = 0 to 30 { 
        x_new = (X * X) - (Y * Y) - 0.7
        y_new = (2.0 * X * Y) + 0.27
        
        # 'blend' acts as a fast GPU if/else for arrays: blend(condition, true, false)
        X = blend(active, x_new, X)
        Y = blend(active, y_new, Y)
        
        mag_sq = (X * X) + (Y * Y)
        escaped = mag_sq > escape_limit
        
        active = blend(escaped, 0.0, active)
    }
}

# 3. Use the results back on the CPU seamlessly!
survivors = sum(active)
print(f"Particles survived: {survivors}")
```

### GPU Block Rules:
* **Math Only:** Use it for numbers, arrays, matrices, basic control flow (`for`, `while`, `if`), and math functions.
* **No Strings or Dictionaries:** You cannot manipulate strings or dictionaries inside the GPU block.
* **Use `blend()` for Array Logic:** If you want to update an array based on a condition, use `blend(condition_array, true_value, false_value)`.

## New in v0.2: Deep Learning & Matrices
Zweriz v0.2 introduces native matrix multiplication using the `@` operator and a built-in `nn` module for neural networks, complete with automatic differentiation and gradient tracking.

```python
batch_size = 2048
epochs = 1000
lr = 0.001

# Initialize Datasets and Weights
X = random.uniform(batch_size, 512, -1.0, 1.0)
Y = random.uniform(batch_size, 10, 0.0, 1.0)
W1 = random.uniform(512, 1024, -0.1, 0.1)

# Tell Zweriz to track gradients for this weight
nn.track(W1)

GPU {
    while epoch < epochs {
        # Native Matrix Multiplication
        Z1 = X @ W1
        # Activation Functions
        A1 = nn.relu(Z1)
        
        # Calculate Loss
        diff = A1 - Y
        loss = diff * diff
        
        # Backpropagation
        nn.backward(loss)
        nn.step(lr)
        nn.zero_grad()
        
        epoch = epoch + 1
    }
}
```

## Built-in Modules
Zweriz comes with several built-in global modules ready to use natively:

* **math**: `math.pow`, `math.floor`, `math.ceil`, `math.round`, `math.transpose`, `math.trapz`, `math.gradient`
* **random**: `random.float()`, `random.randint(min, max)`, `random.uniform(size)`
* **time**: `time.now()`, `time.sleep(seconds)`
* **io**: `io.read(path)`, `io.write(path, text)`, `io.append(path, text)`, `io.exists(path)`
* **os**: `os.cmd(command)`, `os.env(var)`, `os.exit(code)`
* **string**: `string.len(str)`, `string.parse_num(str)`
* **nn (New!)**: Includes standard activations (`nn.relu`, `nn.sigmoid`, `nn.softmax`, `nn.tanh`, `nn.leaky_relu`, `nn.gelu`, `nn.swish`, `nn.softplus`), regularization (`nn.dropout`), and optimizer controls (`nn.track`, `nn.backward`, `nn.step`, `nn.zero_grad`).
* **net (New!)**: Simple and native networking capabilities for web requests and TCP connections (`net.http_get(url)`, `net.http_post(url, body)`, `net.tcp_send(addr, data)`, `net.tcp_listen(addr)`).

## Call for Bug Hunters!
As mentioned, Zweriz is an experimental playground. If you manage to crash the VM, find a memory leak, or discover a script that causes the GPU fallback to trigger incorrectly, please let me know!

**How to report bugs:**
1. Save the exact `.zw` script that caused the issue.
2. Note whether you ran it with or without `--safe`.
3. Submit an issue with the script and the error output.

Thank you for trying Zweriz!
