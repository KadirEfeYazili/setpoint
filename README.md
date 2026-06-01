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

> **Status: early development.** Nothing here is usable yet. The hardware layer and
> `doctor` are being built first. See [Roadmap](#roadmap).

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

Three good tools already exist and none of them talk to each other:

| Tool | Does | Does not |
|---|---|---|
| [gguf-parser-go](https://github.com/gpustack/gguf-parser-go) | Estimates memory from GGUF metadata in seconds | Measure anything |
| [llama-optimus](https://github.com/BrunoArsioli/llama-optimus) | Searches flag combinations with Optuna | Finish quickly, or share the result |
| [llama-swap](https://github.com/mostlygeek/llama-swap) | Orchestrates model processes | Tune anything |

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

## Roadmap

Each phase leaves something usable on its own.

| Phase | Scope | Status |
|---|---|---|
| 0 | Research, ecosystem analysis, positioning | Done |
| 1 | Budgeter, autotuner, doctor, profile format | In progress |
| 2 | Daemon, OpenAI-compatible proxy, TUI, profile sharing, regression sentinel | Planned |
| 3 | Request routing, speculative decoding orchestration, quantization advisor | Planned |
| 4 | Fast model switching, MoE expert cache policy, KV cache tiering | Planned |
| 5 | MCP server, statusline, resource-aware RAG, agent resource API | Planned |
| 6 | Contextual sparsity, learned eviction policies, upstream contribution | Research |

Phase 1 targets llama.cpp only. Ollama and vLLM come in phase 2.

## Scope

setpoint does not replace llama.cpp, Ollama or vLLM. It sits above them and decides
what to tell them. It writes no CUDA kernels and modifies no model files.

## Hardware support

NVIDIA only for now. On AMD, Intel and Apple hardware setpoint reports that it cannot
help rather than producing a number it did not measure.

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

Prior art that shaped the design is credited in [NOTICE](NOTICE). Note that
[llama-moe-cache](https://github.com/ongunm/llama-moe-cache) is AGPL-3.0; its published
ideas and benchmarks informed this project, its code did not.
