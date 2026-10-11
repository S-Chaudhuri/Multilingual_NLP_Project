"""
Build the conda environment for baseline probing and the soft-prompt pipeline.

Run once on a Snellius login node (compute nodes may not have internet):

    module load 2025
    module load Anaconda3/2025.06-1       # see: module spider Anaconda3
    python setup_env.py

Any Python 3 can run this script; it creates a separate Python 3.8 env and
installs everything into it:

    1. conda env `multilingual_nlp` with Python 3.8 (conda-forge only)
    2. requirements_pipeline.txt
    3. UniMorph inflection for Russian / Greek templates: dynet 2.1.2
       (prebuilt wheel, 3.8 only) + a checkout of unimorph_inflect (its pip
       packaging is broken) + its per-language models, downloaded now
       because the library otherwise asks for keyboard input on first use,
       which fails inside a batch job
    4. mBERT and XLM-R weights into the Hugging Face cache
    5. an import / inflection check

Rerunning is safe: an existing env is reused and pip skips what is installed.

Options:
    --env-name NAME       conda env name (default: multilingual_nlp)
    --recreate            delete and rebuild the env
    --no-unimorph         skip step 3 (then run without ru/el)
    --skip-models         skip step 4
"""

import os
import sys
import json
import shutil
import argparse
import subprocess


ROOT = os.path.dirname(os.path.abspath(__file__))
PYTHON_VERSION = "3.8"  # dynet 2.1.2 has a prebuilt wheel only for 3.8
REQUIREMENTS = os.path.join(ROOT, "requirements_pipeline.txt")
UNIMORPH_REPO = "https://github.com/antonisa/unimorph_inflect"
# UniMorph model codes used by scripts/prompt.py for the project languages.
UNIMORPH_MODELS = {"ru": "rus", "el": "ell2"}
HF_MODELS = ["bert-base-multilingual-cased", "xlm-roberta-base"]


def step(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}", flush=True)


def run(cmd, check=True):
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=check)


def find_conda():
    conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if not conda:
        sys.exit(
            "conda not found. On Snellius load it first, e.g.\n"
            "    module load 2023\n"
            "    module spider Anaconda3\n"
            "    module load Anaconda3/<version>"
        )
    return conda


def env_prefix(conda, name):
    out = subprocess.run(
        [conda, "env", "list", "--json"],
        check=True, capture_output=True, text=True,
    ).stdout
    for prefix in json.loads(out)["envs"]:
        if os.path.basename(prefix) == name:
            return prefix
    return None


def env_python(prefix):
    if os.name == "nt":
        return os.path.join(prefix, "python.exe")
    return os.path.join(prefix, "bin", "python")


def install_unimorph(python, prefix):
    """
    Make `import unimorph_inflect` work inside the env.

    `pip install` of this repo is broken: its setup.py installs the
    subpackages as top-level `src` and `utils`, never as `unimorph_inflect`.
    The code expects the repo checkout itself to be a folder named
    `unimorph_inflect`, so clone it into <env>/unimorph_src/unimorph_inflect
    and put <env>/unimorph_src on the env's path with a .pth file.
    """

    # Remove a broken pip install left by an earlier attempt, if any.
    run([python, "-m", "pip", "uninstall", "-y", "unimorph_inflect"], check=False)

    parent = os.path.join(prefix, "unimorph_src")
    checkout = os.path.join(parent, "unimorph_inflect")

    if not os.path.isdir(os.path.join(checkout, ".git")):
        os.makedirs(parent, exist_ok=True)
        clone = run(["git", "clone", "--depth", "1", UNIMORPH_REPO, checkout], check=False)
        if clone.returncode != 0:
            return False

    site_packages = subprocess.run(
        [python, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()

    with open(os.path.join(site_packages, "unimorph_inflect.pth"), "w") as f:
        f.write(parent + "\n")

    return run([python, "-c", "import unimorph_inflect; print('unimorph_inflect importable')"],
               check=False).returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Build the project environment")
    parser.add_argument("--env-name", default="multilingual_nlp")
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--no-unimorph", action="store_true")
    parser.add_argument("--skip-models", action="store_true")
    args = parser.parse_args()

    conda = find_conda()

    # -----------------------------------------------------------------------
    step(f"1. conda env '{args.env_name}' (Python {PYTHON_VERSION})")
    # -----------------------------------------------------------------------

    prefix = env_prefix(conda, args.env_name)

    if prefix and args.recreate:
        # Delete the folder directly: `conda env remove` checks the Anaconda
        # channels' Terms of Service first and fails non-interactively.
        print(f"Removing {prefix}")
        shutil.rmtree(prefix)
        prefix = None

    if prefix is None:
        # conda-forge only: avoids the Anaconda "defaults" channel terms prompt.
        run([conda, "create", "-y", "-n", args.env_name,
             "--override-channels", "-c", "conda-forge",
             f"python={PYTHON_VERSION}", "pip"])
        prefix = env_prefix(conda, args.env_name)
    else:
        print(f"Reusing existing env at {prefix}")

    python = env_python(prefix)
    version = subprocess.run(
        [python, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()

    if version != PYTHON_VERSION:
        sys.exit(
            f"{prefix} has Python {version}, expected {PYTHON_VERSION}. "
            f"Rebuild it with: python setup_env.py --recreate"
        )

    # pip must install into the env, never into ~/.local.
    os.environ["PYTHONNOUSERSITE"] = "1"
    pip = [python, "-m", "pip", "install"]

    # -----------------------------------------------------------------------
    step("2. Python packages (requirements_pipeline.txt)")
    # -----------------------------------------------------------------------

    run(pip + ["--upgrade", "pip"])
    run(pip + ["-r", REQUIREMENTS])

    # -----------------------------------------------------------------------
    step("3. UniMorph inflection (Russian / Greek templates)")
    # -----------------------------------------------------------------------

    unimorph_ok = False

    if args.no_unimorph:
        print("Skipped (--no-unimorph): run without ru/el, e.g. LANGS=en,nl,ko")
    else:
        # --only-binary: never fall back to compiling dynet, which fails.
        dynet = run(pip + ["--only-binary=dynet", "dynet==2.1.2", "requests", "protobuf"],
                    check=False)
        if dynet.returncode == 0:
            unimorph = install_unimorph(python, prefix)
            if unimorph:
                # Download models now: on first use the library asks for
                # keyboard input, which fails inside a batch job.
                download = (
                    "from unimorph_inflect.utils.resources import download, DEFAULT_MODEL_DIR\n"
                    "import os\n"
                    f"for code in {sorted(UNIMORPH_MODELS.values())!r}:\n"
                    "    if not os.path.isdir(os.path.join(DEFAULT_MODEL_DIR, code)):\n"
                    "        download(code, resource_dir=DEFAULT_MODEL_DIR, force=True)\n"
                    "    print('model ready:', os.path.join(DEFAULT_MODEL_DIR, code))\n"
                )
                unimorph_ok = run([python, "-c", download], check=False).returncode == 0

        if not unimorph_ok:
            print("\nWARNING: UniMorph setup failed; run without ru/el (LANGS=en,nl,ko).")

    # -----------------------------------------------------------------------
    step("4. Hugging Face model weights")
    # -----------------------------------------------------------------------

    if args.skip_models:
        print("Skipped (--skip-models)")
    else:
        run([python, "-c",
             "from transformers import AutoTokenizer, AutoModelForMaskedLM\n"
             f"for name in {HF_MODELS!r}:\n"
             "    AutoTokenizer.from_pretrained(name)\n"
             "    AutoModelForMaskedLM.from_pretrained(name)\n"
             "    print('cached', name)\n"])

    # -----------------------------------------------------------------------
    step("5. Check")
    # -----------------------------------------------------------------------

    check = (
        "import sys, torch, transformers\n"
        f"sys.path.insert(0, {os.path.join(ROOT, 'scripts')!r})\n"
        f"sys.path.insert(0, {os.path.join(ROOT, 'pipeline_v1')!r})\n"
        "import prompt, probe, prompt_tuning_data\n"
        "print('python      ', sys.version.split()[0])\n"
        "print('torch       ', torch.__version__, '| CUDA visible:', torch.cuda.is_available(),\n"
        "      '(False is normal on a login node)')\n"
        "print('transformers', transformers.__version__)\n"
        "print('UniMorph    ', prompt.HAS_UNIMORPH)\n"
        "if prompt.HAS_UNIMORPH:\n"
        "    from unimorph_inflect import inflect\n"
        "    print('inflect rus ', inflect('город', 'N;GEN;SG', language='rus'))\n"
        "    print('inflect ell2', inflect('πόλη', 'N;GEN;SG', language='ell2'))\n"
    )
    # probe.py and prompt.py read data/ with relative paths; the project's data
    # files are UTF-8 (matters on Windows, where the default encoding is not).
    subprocess.run([python, "-c", check], check=True, cwd=ROOT,
                   env=dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8"))

    langs = "en,nl,ru,el,ko" if unimorph_ok else "en,nl,ko"
    print(f"""
Environment ready. Before submitting jobs:

    export PYTHON={python}

Baseline probing:     sbatch baseline_probe.slurm{'' if unimorph_ok else '   (with LANGS=' + langs + ')'}
Ablation/ensembling:  bash run_ablation.sh{'' if unimorph_ok else '       (with LANGS=' + langs + ')'}
""")


if __name__ == "__main__":
    main()
