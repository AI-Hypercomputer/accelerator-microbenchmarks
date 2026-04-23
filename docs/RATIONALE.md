# Rationale for the New `jax-benchmarks` Repository

This document outlines the motivation for creating the `jax-benchmarks`
repository, the limitations of previous benchmarking approaches, and how this
new framework addresses those gaps to provide a more robust, scalable, and
insightful performance analysis platform for JAX.

## Limitations of Previous Approaches

Prior to `jax-benchmarks`, benchmarking efforts often suffered from several
drawbacks:

1.  **Ad-hoc Scripts:** Benchmarks were frequently implemented as standalone
    scripts, leading to code duplication, inconsistent methodologies, and
    difficulty in comparing results across different operations or hardware.
2.  **Lack of Standardization:** There was no unified structure for defining
    benchmarks, running experiments, or collecting metrics. This made it hard to
    enforce best practices like proper warm-up, statistical significance, or
    power-aware measurements.
3.  **Difficult Configuration:** Managing and running benchmarks across many
    different shapes, sizes, and hardware configurations was manual and
    error-prone, often involving hardcoded parameters or complex shell script
    loops.
4.  **Limited Analysis:** Scripts typically only reported raw execution times.
    Deeper insights, such as roofline efficiency, arithmetic intensity, or
    memory bandwidth utilization, required significant manual post-processing
    and external calculations.
5.  **Poor Extensibility:** Adding new benchmarks or modifying existing ones
    often required significant code changes and understanding of bespoke script
    logic, hindering community contributions and rapid iteration.
6.  **Scalability Issues:** Running and managing hundreds or thousands of
    benchmark configurations (common in large-scale hardware/software co-design
    studies) was practically infeasible.

## The `jax-benchmarks` Approach

This repository introduces a structured and extensible framework designed to
address the above limitations:

1.  **Modular & Registry-Based:** Benchmarks are classes inheriting from
    `BaseBenchmark`, automatically registered and discoverable. This promotes
    code reuse and makes adding new benchmarks as simple as creating a new file.
2.  **Standardized Orchestration:** The `BaseBenchmark` class enforces a
    consistent lifecycle: `setup`, `generate_inputs`, `run_op`, and
    `calculate_metrics`. It handles warm-up, timing, and integration with
    analysis tools.
3.  **Hierarchical & Bulk Configuration:**
    *   **YAML Configs:** Cleanly define benchmarks, parameters, and hardware
        specs.
    *   **Sweeps:** Automatically run Cartesian products of parameters within
        YAML.
    *   **CSV/Sheets Ingestion:** Seamlessly load thousands of shapes from
        external datasheets, perfect for large studies.
4.  **Built-in Roofline Analysis:** Automatic calculation of arithmetic
    intensity, roofline ceilings (compute and memory), and efficiency
    percentages, providing immediate insights into performance bottlenecks.
5.  **Analytical & Trace-Based Roofline:** Offers both high-level formula-based
    roofline and deep dives into compiled Jaxprs using
    `jax.experimental.roofline`.
6.  **Clear Documentation:** Includes `README.md` for users and `DEVELOPERS.md`
    for contributors, lowering the barrier to entry.
7.  **Installable Package:** Facilitates easy distribution and usage within
    other projects or CI/CD pipelines with a simple `pip install` and
    `jax-bench` CLI.

## Addressing the Gaps

-   **Frictionless Scaling:** From single runs to thousands of configurations
    via YAML and CSV.
-   **Actionable Insights:** Beyond latency, now includes TFLOPS, GB/s,
    intensity, and roofline efficiency out-of-the-box.
-   **Reduced Toil:** Automates the tedious aspects of setup, execution, and
    data collection.
-   **Community Ready:** Standardized structure encourages contributions and
    sharing of benchmarks.
-   **High-Fidelity:** Supports power-aware runs (`min_duration_s`) for
    realistic performance under load.

In summary, `jax-benchmarks` provides a professional, scalable, and insight-rich
platform for JAX performance analysis, moving beyond one-off scripts to a
sustainable and extensible ecosystem.
