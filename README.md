# Zweriz (v0.1)
A fast, dynamic programming language with native, seamless GPU acceleration.

Zweriz is a modern scripting language designed to make high-performance computing as easy as writing a simple script. Whether you are building a quick math tool or simulating millions of particles, Zweriz writes like a standard dynamic language but gives you the power to offload heavy math directly to your NVIDIA GPU with a single block of code.

## A Note from the Creator
Hi! I'm the solo teenage developer behind Zweriz. This is the Initial v0.1 Release. Because I am building this on my own and don't have the budget to hire a QA team, there are bound to be undiscovered bugs and edge cases out there in the wild. I am releasing the compiled binary so people can play with it, test it, and push it to its limits.

**Note:** The Interactive REPL is currently in its early stages and has some known syntax parsing quirks compared to running a `.zw` file directly. I apologize for any inconvenience while I iron those out!

If you are interested in languages, compilers, or GPU tech, I would massively appreciate volunteer bug hunters! Try to break it, and let me know what you find so we can make v0.2 even better.

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

### Functions
```python
fn calculate_area(width, height) {
    return width * height
}

area = calculate_area(10, 5)
```

### Error Handling
```python
try {
    # Risky code
    if missing_data {
        throw "Data is missing!"
    }
} catch (err) {
    print(f"Caught an error: {err}")
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
To keep the GPU lightning-fast, there are a few rules for what goes inside a `GPU {}` block:

* **Math Only:** Use it for numbers, arrays, matrices, basic control flow (`for`, `if`), and math functions (`sin`, `cos`, `sqrt`, `abs`).
* **No Strings or Dictionaries:** You cannot manipulate strings or dictionaries inside the GPU block. Extract dictionary values to scalar variables before entering the GPU block.
* **Use `blend()` for Array Logic:** If you want to update an array based on a condition, use `blend(condition_array, true_value, false_value)`.
* **Custom Functions:** You can call custom `fn` functions inside a GPU block, provided the function itself only contains GPU-safe math!

## Built-in Modules
Zweriz comes with several built-in global modules ready to use natively:

* `math`: `math.pow`, `math.floor`, `math.ceil`, `math.round`, `math.transpose`, `math.trapz`, `math.gradient`
* `random`: `random.float()`, `random.randint(min, max)`, `random.uniform(size)`
* `time`: `time.now()`, `time.sleep(seconds)`
* `io`: `io.read(path)`, `io.write(path, text)`, `io.append(path, text)`, `io.exists(path)`
* `os`: `os.cmd(command)`, `os.env(var)`, `os.exit(code)`
* `string`: `string.len(str)`, `string.parse_num(str)`

## Call for Bug Hunters!
As mentioned, V0.1 is an experimental playground. If you manage to crash the VM, find a memory leak, or discover a script that causes the GPU fallback to trigger incorrectly, please let me know!

**How to report bugs:**
1. Save the exact `.zw` script that caused the issue.
2. Note whether you ran it with or without `--safe`.
3. Submit an issue with the script and the error output.

Thank you for trying Zweriz!
