# Brainrust 🧠

An interpreter and compiler for the Brainfuck language.

## Prerequisites

You need a LLVM 16.0.* installation on your system. On Windows, you may need to
build LLVM from source, since the pre-built binaries do not include everything
you need (see below).

### Building LLVM from Source on Windows

You will have to have Visual Studio installed.

```bash
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout tags/llvmorg-16.0.4
mkdir build
cd build
cmake -DLLVM_ENABLE_PROJECTS=clang -A x64 -Thost=x64 ..\llvm
```

This creates a file called `LLVM.sln`. You can open that file in Visual Studio,
choose the desired build type (e.g. "Release") and build the `BUILD_ALL` target.

Make sure to add the `/bin` folder of the resulting build artifacts to `PATH`.
Verify that it worked:

```bash
clang --version
```

This should yield something like this:

```bash
clang version 16.0.4
Target: x86_64-pc-windows-msvc
Thread model: posix
InstalledDir: C:\dev\llvm-project\build\Release\bin
```

## Getting Started

Run the program using cargo. Pass `--help` to see the available options.

```bash
cargo run -- --help
```

To run an example program using the built-in interpreter, type:

```bash
cargo run -- -r programs/hello_world.b
```

To compile a program, type:

```bash
cargo run -- -o hello_world.exe programs/hello_world.b
```

The compiler uses `clang` for linking. Make sure that `clang` is available in
the `PATH`.
