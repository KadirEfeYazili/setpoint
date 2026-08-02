```
███████╗███████╗████████╗██████╗  ██████╗ ██╗███╗   ██╗████████╗
██╔════╝██╔════╝╚══██╔══╝██╔══██╗██╔═══██╗██║████╗  ██║╚══██╔══╝
███████╗█████╗     ██║   ██████╔╝██║   ██║██║██╔██╗ ██║   ██║
╚════██║██╔══╝     ██║   ██╔═══╝ ██║   ██║██║██║╚██╗██║   ██║
███████║███████╗   ██║   ██║     ╚██████╔╝██║██║ ╚████║   ██║
╚══════╝╚══════╝   ╚═╝   ╚═╝      ╚═════╝ ╚═╝╚═╝  ╚═══╝   ╚═╝
```

**Measurement-driven configuration for local LLM inference.**

`setpoint` finds the configuration your hardware can actually hold, by measuring it
instead of guessing, and remembers the answer.

> **Status: early development.** All seven commands run, and have been used to measure
> real hardware. Nothing is published to a package index yet. See [Roadmap](#roadmap).

---

## The problem

Running a model locally means choosing values for `-ngl`, `-ot`, `--n-cpu-moe`, `-c`,
`-b`, `-ub`, `--flash-attn` and a dozen more flags. The optimum depends on the model,
the GPU, the driver and how you actually use it. Today the only method is trial and
error.

It gets worse when the guess is slightly wrong. On Windows, a model that does not fit
in VRAM does not fail: the driver quietly backs the overflow with system RAM. The
layers still report as GPU-resident while running 5 to 20 times slower, and nothing
tells you. People conclude their GPU is too weak when the real problem is that nobody
is measuring.

Three kinds of tool already exist, and none of them connect:

| Kind | Does | Does not |
|---|---|---|
| Static estimators | Compute memory requirements from model metadata in seconds | Measure anything |
| Flag search tools | Explore parameter combinations by benchmarking | Finish quickly, or produce a result anyone else can reuse |
| Process orchestrators | Start, stop and route between model servers | Tune anything |

There is no shortage of mechanism. What is missing is policy: something that decides
what the numbers should be, on this machine, for this model, and can prove it.

## What setpoint does

```
measure  ->  compare against the setpoint  ->  apply  ->  measure again
```

- **Budget.** Reads GGUF metadata, computes the VRAM you can actually use (driver
  overhead, other processes, fragmentation headroom) and models the KV cache, so the
  tradeoff between context length and resident layers is visible before you run anything.
- **Tune.** Seeds a search from the static estimate, then measures. Successive halving
  followed by coordinate descent, in minutes rather than hours.
- **Doctor.** Scans for the traps that silently cost throughput: CUDA runtime mismatch,
  VRAM spill into shared system memory, thermal and power throttling, PCIe link state.
- **Profile.** Stores the result keyed by a hardware and model signature, together with
  the measurement that justifies it. A profile that cannot show what it beat is not
  written.

### Measurement discipline

A tuning tool is only worth the trust in its numbers, so the rules are strict:

- The first run is a warmup and never counted.
- Every configuration is measured at least three times; the median is reported and the
  interquartile range is kept.
- If IQR over median exceeds 5 percent the result is marked unreliable and is not
  written to a profile.
- Thermal and throttle state is read before every measurement.
- Every profile records the baseline it was compared against.

## Install

Not published yet. From a checkout:

```bash
uv pip install -e .
setpoint doctor
```

Requires Python 3.10+ and, for GPU checks, an NVIDIA driver.

## Budgeting

`budget` runs nothing and measures no throughput. It reads the model header, reads what
the driver reports as free, and works out where the split has to fall.

```
$ setpoint budget qwen3:8b -c 8192

model
  qwen3:8b   qwen3 8.2B dense, Q4_K_M, 36 blocks

vram  NVIDIA GeForce GTX 1650
  total                   4.00 GiB
  free                    3.87 GiB   measured
  fragmentation          -0.12 GiB
  runtime allowance      -0.19 GiB   estimate; `setpoint tune` measures it
  safe ceiling            3.57 GiB

need  at 8192 tokens
  weights                 4.86 GiB
  KV cache f16/f16        1.12 GiB   144.0 KiB per token
  total                   5.99 GiB

plan
  -ngl 24                 24 of 36   blocks on the GPU
  on cpu                  2.53 GiB   42% of the model

instead
  -ctk q8_0 -ctv q8_0      -ngl 27   frees 0.53 GiB
  -c 4096                  -ngl 27   frees 0.56 GiB
```

It exits 0 when the request fits entirely on the GPU and 1 when part of it has to stay
on the CPU, so it composes into scripts. `--json` emits the whole plan, including the
candidate configurations the tuner will start from. The model argument takes a path to
a `.gguf` file or the name of a model you already have locally.

Two figures are honest about what they are. The runtime allowance covers the driver
context and compute buffers of a process that has not started yet, so it is a stated
default rather than a measurement. And where an architecture's KV cache does not follow
the usual per-head layout, setpoint says it cannot size it instead of printing a number
it did not derive.

## Tuning

`tune` starts from the budget, screens the candidates cheaply, then walks one parameter
at a time, measuring each move. The result is a profile keyed to this machine.

```
$ setpoint tune qwen2.5:3b -c 1024

plan
  seeded from the budget: -ngl 37, 36 of 36 blocks
  target 1024 tokens, optimising for speed
  measuring on Vulkan0 -- NVIDIA GeForce GTX 1650

search
  warmup   -ngl 99      39.12 kept
  screen   -ngl 35      47.37 kept
  screen   -ngl 37      49.40 kept                  best of the seeds
  descend  -ngl 36      49.22 no better             -ngl 36
  descend  -ngl 37      53.66 improved              -ub 128
  descend  -ngl 35      48.77 no better             -ngl 35
  confirm  -ngl 37      52.77 kept
  baseline -ngl 99      50.77 kept

result
  -ngl 37, -ub 128, flash attention on
  52.77 tok/s, spread 0.4%, peak vram 2560 MiB
  speedup 1.04x over the llama.cpp default
```

Three details in that trace are the discipline showing through. The first measurement is
thrown away, because a GPU reads low while its clocks ramp and the readings climb over
the first minute of a session. The baseline is measured last, next to the winner, so the
two numbers that form the speedup claim share a thermal state -- measured first, the
baseline was the coldest reading of the session and inflated every result. And the
accelerator is resolved by name and printed, because device ids are positional and a
reboot can renumber them: an integrated GPU will happily accept the work and report a
number that describes itself rather than the card the profile claims.

`Ctrl-C` keeps the best configuration measured so far. A result whose spread is too wide
is not written to a profile at all.

## Applying and re-checking

`run` starts llama-server with the profile measured for this machine. Anything after `--`
goes to the server untouched.

```
$ setpoint run qwen2.5:3b -- --port 9090

profile
  32ca34ae9eef85a1   Qwen2.5 3B Instruct Q4_K_M
  measured               52.77 t/s   2026-09-08T18:34:19Z
```

`bench` re-measures the stored configuration and says whether the profile still describes
the machine. A driver update, a backend update or a new card breaks the signature; a
quieter or busier machine shows up here as drift.

```
$ setpoint bench qwen2.5:3b

  profile claims         52.77 t/s   2026-09-08T18:34:19Z
  measured now           51.76 t/s   spread 1.4%
  difference                         -1.9%

the profile still holds
```

It exits 1 when the difference leaves the tolerance, which makes it usable from a
scheduler.

## Roadmap

Each phase leaves something usable on its own.

| Phase | Scope | Status |
|---|---|---|
| 0 | Research, ecosystem analysis, positioning | Done |
| 1 | Budgeter, autotuner, doctor, profile format | In progress |
| 2 | Daemon, OpenAI-compatible proxy, TUI, profile sharing, regression sentinel | Planned |
| 3 | Request routing, speculative decoding orchestration, quantization advisor | Planned |
| 4 | Fast model switching, MoE expert cache policy, KV cache tiering | Planned |
| 5 | Knowledge layer: measured chunking, retrieval policy, embedding placement, search | Planned |
| 6 | Reasoning layer: MCP server, statusline, agent resource API | Planned |
| 7 | Contextual sparsity, learned eviction policies, upstream contribution | Research |

Phase 1 targets llama.cpp only. Ollama and vLLM come in phase 2.

## Architecture

setpoint decides across four layers, and all of them settle against the same budget.

| Layer | Decides |
|---|---|
| Hardware | How much room is actually available |
| Execution | What runs, with which settings, and what stays resident |
| Knowledge | What is retrieved and how it is represented |
| Reasoning | How to proceed, and when to stop |

What ties them together is that every one of these choices has a measurable resource
cost and a quality or latency tradeoff. On constrained hardware something has to make
that tradeoff deliberately.

## Providers

Nothing requires an account, a key or an internet connection. Every external capability
sits behind a provider interface, and the default provider is always local and keyless.
Bring your own inference endpoint, search backend, embedding model, reranker or storage
if you want one; keys stay on your machine and are never written to profiles or logs.
No provider is privileged.

## Scope

setpoint does not replace llama.cpp, Ollama or vLLM. It sits above them and decides
what to tell them. It writes no CUDA kernels and modifies no model files.

## Hardware support

NVIDIA only for now. On AMD, Intel and Apple hardware setpoint reports that it cannot
help rather than producing a number it did not measure.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
