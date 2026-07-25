#!/usr/bin/env bash
# Shared repo-local defaults for the math-with-code bring-up.
#
# Source this file from the NeMo-RL repo checkout. Callers may override any of
# the exported variables before sourcing it.

MATH_CODE_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

MATH_CODE_ARCH="${MATH_CODE_ARCH:-$(uname -m)}"
case "$MATH_CODE_ARCH" in
    aarch64|x86_64) ;;
    *)
        printf 'Unsupported math-code architecture: %s\n' "$MATH_CODE_ARCH" >&2
        return 1 2>/dev/null || exit 1
        ;;
esac

MATH_CODE_ARTIFACT_ROOT="${MATH_CODE_ARTIFACT_ROOT:-$MATH_CODE_PROJECT_ROOT/.artifacts/$MATH_CODE_ARCH}"
case "$MATH_CODE_ARTIFACT_ROOT" in
    /*) ;;
    *) MATH_CODE_ARTIFACT_ROOT="$MATH_CODE_PROJECT_ROOT/$MATH_CODE_ARTIFACT_ROOT" ;;
esac
MATH_CODE_SIF_PATH="${MATH_CODE_SIF_PATH:-$MATH_CODE_ARTIFACT_ROOT/math-code-sif_py312-$MATH_CODE_ARCH.sif}"
case "$MATH_CODE_SIF_PATH" in
    /*) ;;
    *) MATH_CODE_SIF_PATH="$MATH_CODE_PROJECT_ROOT/$MATH_CODE_SIF_PATH" ;;
esac
# Task TOMLs use a project-relative reference for the default SIF. The custom
# Singularity environment resolves it against this project rather than the
# Harbor process cwd. External overrides remain absolute.
case "$MATH_CODE_SIF_PATH" in
    "$MATH_CODE_PROJECT_ROOT"/*)
        _MATH_CODE_DEFAULT_SIF_REFERENCE="${MATH_CODE_SIF_PATH#"$MATH_CODE_PROJECT_ROOT"/}"
        ;;
    *)
        _MATH_CODE_DEFAULT_SIF_REFERENCE="$MATH_CODE_SIF_PATH"
        ;;
esac
MATH_CODE_SIF_REFERENCE="${MATH_CODE_SIF_REFERENCE:-$_MATH_CODE_DEFAULT_SIF_REFERENCE}"
# NeMo-RL images set NEMO_GYM_VENV_DIR=/opt/gym_venvs. That container-local
# default is unsuitable for a multi-node bring-up, so do not treat it as an
# explicit math-code override. Use the project-specific variable when a custom
# shared location is genuinely required.
NEMO_GYM_VENV_DIR="${MATH_CODE_NEMO_GYM_VENV_DIR:-$MATH_CODE_ARTIFACT_ROOT/gym_venvs}"

export MATH_CODE_ARCH
export MATH_CODE_PROJECT_ROOT
export MATH_CODE_ARTIFACT_ROOT
export MATH_CODE_SIF_PATH
export MATH_CODE_SIF_REFERENCE
export MATH_CODE_NEMO_GYM_VENV_DIR
export NEMO_GYM_VENV_DIR

unset _MATH_CODE_DEFAULT_SIF_REFERENCE
