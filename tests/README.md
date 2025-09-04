# Shop-R1 Testing Suite

This directory contains all testing scripts and utilities for the Shop-R1 implementation.

## Quick Start

### 1. Basic Environment Test
```bash
python tests/quick_test.py
```
This validates that the environment loads and rewards are computed correctly.

### 2. Local Testing (if you have verifiers installed)
```bash
./tests/test_locally.sh
```

### 3. Remote Testing on Prime Intellect Pod
See `FRESH_POD_SETUP.md` for complete step-by-step instructions.

## Test Files

- `quick_test.py` - Quick validation of environment and rewards
- `test_locally.sh` - Local testing script (requires Python 3.9+)
- `test_remote.sh` - Full remote testing suite for Prime Intellect pods
- `test_standalone.py` - Standalone tests without verifiers dependency

## Running Tests on Prime Intellect

1. SSH into your pod
2. Clone the repository
3. Run `python tests/quick_test.py` first
4. Follow the commands in `docs/internal/pod_commands.txt`