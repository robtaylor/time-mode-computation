# Claude Development Guidelines for Time-Mode Computation

This document provides guidelines for Claude or other AI assistants working on this codebase.

## Project Overview

This is a Python simulation framework for time-mode analog computation circuits, implementing concepts from recent research papers on ultra-low energy neuromorphic computing.

## Development Requirements

### Pre-commit Checklist

Before committing any changes, ensure:

1. **All tests pass**:
   ```bash
   pdm run test
   ```

2. **Code passes linting**:
   ```bash
   pdm run ruff check .
   pdm run ruff format --check .
   ```

3. **Examples run successfully**:
   ```bash
   pdm run example-vmm
   pdm run example-neural
   ```

### Code Style

- **Python version**: 3.11
- **Type hints**: Use Python 3.11 style type hints (e.g., `list[str]` instead of `List[str]`)
- **Formatting**: Use ruff formatter with line length of 100
- **Imports**: Organized by ruff/isort standards
- **Docstrings**: Google style with proper formatting (no trailing whitespace)

### Key Conventions

1. **Signal Representation**:
   - Use `TimeSignal` for single-ended signals
   - Use `DifferentialTimeSignal` for signed values
   - Pulse width encoding is the default

2. **Energy Metrics**:
   - Report in fJ/Op (femtojoules per operation)
   - Include comparisons to literature values

3. **VMM Operations**:
   - Single-quadrant for unsigned values
   - Four-quadrant for signed values using differential signaling

### Testing Guidelines

- Run examples with `MPLBACKEND=Agg` for headless environments
- Ensure VMM computations match numpy reference within 10% relative tolerance
- Test both single and four-quadrant operations

### CI/CD

- GitHub Actions runs on push and PR to main branch
- Tests run on Ubuntu and macOS with Python 3.11
- Linting must pass (no auto-fixes in CI)
- Type checking is currently informational only

### Common Tasks

#### Run all checks locally:
```bash
# Install dependencies
pdm install

# Run tests
pdm run test

# Check linting
pdm run lint

# Format code
pdm run ruff format .

# Run specific example
pdm run example-vmm
pdm run example-neural
```

#### Fix common issues:
```bash
# Auto-fix linting issues
pdm run ruff check . --fix

# Format code
pdm run ruff format .
```

### Architecture Notes

1. **Core modules**:
   - `core.py`: Signal representations and current sources
   - `blocks.py`: Basic building blocks (MSMV, FWPG, etc.)
   - `vmm.py`: Vector-matrix multiplication
   - `neural.py`: Neural network layers and architectures
   - `visualization.py`: Analysis and plotting tools

2. **Key abstractions**:
   - Time-encoded signals carry analog information in pulse widths
   - Current sources model programmable weights
   - Charge pumps integrate currents over time
   - Phase-based operation (Phase I: compute, Phase II: output)

### Performance Targets

Based on literature:
- Target: <10 fJ/Op for large arrays
- Current simulation: ~500 fJ/Op (simplified model)
- Bavandpour et al. (2019): 7 fJ/Op @ 55nm

### Future Work

- [ ] Implement training algorithms for time-domain networks
- [ ] Add noise modeling and variation analysis
- [ ] Implement full convolutional layer with time-multiplexing
- [ ] Add hardware synthesis support
- [ ] Create benchmarking suite against other frameworks

## Important Notes

- **Always ensure tests and linting pass before committing**
- **Follow existing code patterns and conventions**
- **Document any significant architectural changes**
- **Keep examples working and well-documented**
- **Always ensure `pdm check` passes before pushing**

