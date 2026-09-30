```
────────────────────────────────────────────────────────────────
███████╗███████╗████████╗██████╗  ██████╗ ██╗███╗   ██╗████████╗
██╔════╝██╔════╝╚══██╔══╝██╔══██╗██╔═══██╗██║████╗  ██║╚══██╔══╝
███████╗█████╗     ██║   ██████╔╝██║   ██║██║██╔██╗ ██║   ██║   
╚════██║██╔══╝     ██║   ██╔═══╝ ██║   ██║██║██║╚██╗██║   ██║   
███████║███████╗   ██║   ██║     ╚██████╔╝██║██║ ╚████║   ██║   
╚══════╝╚══════╝   ╚═╝   ╚═╝      ╚═════╝ ╚═╝╚═╝  ╚═══╝   ╚═╝   
────────────────────────────────────────────────────────────────
measurement-driven configuration for local inference
```

`setpoint` finds the configuration your hardware can actually hold, by measuring it
instead of guessing, and remembers the answer.

> **Status: early development.** All fifteen commands run, and have been used to
> measure real hardware. Nothing is published to a package index yet.
> See [Roadmap](#roadmap).

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
- **Apply.** Turns the measured profiles into a runner's configuration, with the
  measurement behind each entry kept as a comment, so the settings that reach the engine
  are the ones that were measured.
- **Watch.** Re-measures a profile and decides statistically whether the machine has
  slowed, separating a real regression from a busy afternoon, and says what changed.
- **Choose.** Costs a request against every measured model, counting what it takes to
  load one that is not already running; measures whether speculative decoding pays on
  this card and for which kind of work; and compares local quantizations on speed, VRAM
  and divergence from the most precise copy present.

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

Requires Python 3.10+ and, for GPU checks, an NVIDIA driver. The core has two
dependencies. `setpoint panel` needs one more and is an optional extra:

```bash
uv pip install -e ".[tui]"
```

Measuring throughput needs a llama.cpp build on `PATH`. If the binaries live elsewhere,
point at them:

```bash
export SETPOINT_LLAMA_BENCH=/path/to/llama-bench
export SETPOINT_LLAMA_SERVER=/path/to/llama-server
```

## Commands

`MODEL` is a path to a `.gguf` file, or the name of one already on this machine.
Names are matched against the directories models are kept in; add your own with
`SETPOINT_MODELS_DIR`.

```bash
setpoint doctor                      # scan for traps that silently cost throughput
setpoint hardware                    # what the driver and the OS report right now
setpoint budget MODEL -c N           # where the split has to fall, running nothing
setpoint tune MODEL -c N             # measure a configuration and write a profile
setpoint bench MODEL                 # re-measure a profile, say whether it still holds
setpoint run MODEL -- ARGS           # start llama-server with the measured profile
setpoint chat MODEL                  # talk to it, with each answer's cost beside it
setpoint route --tokens N            # which measured model answers, switch cost included
setpoint spec MODEL                  # whether speculative decoding pays here, per workload
setpoint quant MODEL -f CORPUS       # compare local quantizations: speed, VRAM, drift
setpoint export --target NAME        # runner configuration from the measured profiles
setpoint top                         # live: VRAM, shared memory, throttle
setpoint status                      # one-shot machine state
setpoint profile list|show|path|export|import
setpoint panel                       # one screen for everything measured
```

Exit codes are part of the contract: `0` healthy, `1` a problem was found, `2` setpoint
could not complete the check. Data goes to stdout and diagnostics to stderr, so every
command composes with `jq` and shell pipelines.

`--json` is accepted by `doctor`, `hardware`, `budget`, `tune`, `bench`, `route`,
`spec`, `quant`, `status` and `profile`. `export` writes its own format, `top`, `panel`
and `chat` are screens, and `run` hands over to the server.

A first session, in order:

```bash
setpoint doctor                      # fix what it reports before measuring anything
setpoint budget qwen2.5:3b -c 4096   # see the tradeoff
setpoint tune qwen2.5:3b -c 4096     # measure it, write the profile
setpoint run qwen2.5:3b              # use it
setpoint chat qwen2.5:3b             # or talk to it and watch what it costs
setpoint bench qwen2.5:3b            # later: is it still true
```

## Budgeting

`budget` runs nothing and measures no throughput. It reads the model header, reads what
the driver reports as free, and works out where the split has to fall.

```
$ setpoint budget qwen3:8b -c 8192

model
  qwen3:8b   qwen3 8.2B dense, Q4_K_M, 36 blocks
  ~/.ollama/models/blobs/sha256-a3de86cd1..0b8e686f   4.87 GiB
  trained context 40960, 32 heads over 8 KV heads

vram  NVIDIA GeForce GTX 1650
  total                   4.00 GiB
  free                    3.06 GiB   measured now
  fragmentation          -0.25 GiB
  runtime allowance      -0.34 GiB   calibrated on Vulkan, GTX 1650, two vocabularies
  safe ceiling            2.48 GiB

need  at 8192 tokens
  weights                 4.86 GiB
  KV cache f16/f16        1.12 GiB   144.0 KiB per token
  total                   5.99 GiB

plan
  -ngl 17                 17 of 36   blocks on the GPU
  on gpu                  2.45 GiB   weights 1.92 GiB + cache 0.53 GiB
  next block needs        0.15 GiB   desktop usage drifting by this much moves the plan
  on cpu                  3.53 GiB   59% of the model

instead
  -ctk q8_0 -ctv q8_0      -ngl 19   frees 0.53 GiB
      a quantized KV cache, at the same context
  -c 4096                  -ngl 19   frees 0.56 GiB
      a shorter context, at the same cache precision

  note: What stays on the CPU (3.53 GiB) is a large share of the 3.69 GiB of RAM free
  right now, and reading the model will cache up to 4.87 GiB more. Close something
  before measuring.
```

It exits 0 when the request fits entirely on the GPU and 1 when part of it has to stay
on the CPU, so it composes into scripts. `--json` emits the whole plan, including the
candidate configurations the tuner will start from. The model argument takes a path to
a `.gguf` file or the name of a model you already have locally.

Three lines are honest about what they are. The runtime allowance covers the driver
context and compute buffers of a process that has not started yet; it is computed from
constants fitted on measured peaks, and the output names what they were fitted on, so a
different backend is a reason to distrust it. `next block needs` is there because the
budget is a single reading and a desktop's own VRAM use moves while you read it. And
where an architecture's KV cache does not follow the usual per-head layout, setpoint
says it cannot size it instead of printing a number it did not derive.

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

## Talking to it

`chat` starts the engine with the measured configuration and talks to it. There is no
proxy and no endpoint of its own: the server exists while the conversation does.

```
$ setpoint chat gemma3:1b

model
  a748e45f1a843e88   gemma3 Q4_K_M
  running on Vulkan1 -- NVIDIA GeForce GTX 1650
  measured               86.71 t/s   2026-09-09T12:23:24Z
  context                     4096
  speculator            ngram-simple   measured to help here

you  Write a short paragraph about why measurement beats estimation.

model
Precise measurement provides a robust foundation for understanding and comparison...
  61 tokens   81.9 t/s (-5% on the profile)   peak 1509 MiB
```

Every answer carries what it cost and how that compares with the profile, which is the
only way to notice that a machine has drifted while you are using it rather than when
you next run `bench`. The accelerator is named for the same reason it is named during
tuning: device ids are positional, and a reboot can renumber them.

The context is kept by asking the server's own tokenizer how long the conversation is,
not by estimating from character counts, and the oldest exchanges are dropped in pairs
when it no longer fits.

## Roadmap

Each phase leaves something usable on its own.

| Phase | Scope | Status |
|---|---|---|
| 0 | Research, ecosystem analysis, positioning | Done |
| 1 | Budgeter, autotuner, doctor, profile format | Done |
| 2 | Runner configuration from measured profiles, live telemetry, profile sharing, regression sentinel | Done |
| 3 | Request routing, speculative decoding orchestration, quantization advisor, panel | Done |
| 4 | Fast model switching, MoE expert cache policy, KV cache tiering | In progress |
| 5 | Knowledge layer: measured chunking, retrieval policy, embedding placement, search | Planned |
| 6 | Reasoning layer: MCP server, statusline, agent resource API | Planned |
| 7 | Contextual sparsity, learned eviction policies, upstream contribution | Research |

llama.cpp is the only engine driven so far. Model files are found by searching the
directories they are kept in, so a name works wherever the file already lives; point
`SETPOINT_MODELS_DIR` at your own directory to add one. Other engines sit behind the
same provider interface and are not done.

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

setpoint does not replace an inference engine. It sits above one and decides what to
tell it. It writes no CUDA kernels, modifies no model files, and serves no requests of
its own: it reads the files that are already on the disk and drives the engine directly.

## Hardware support

NVIDIA only for now. On AMD, Intel and Apple hardware setpoint reports that it cannot
help rather than producing a number it did not measure.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
