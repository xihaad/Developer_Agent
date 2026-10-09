"""Build submission.zip for the Gemma 4 Developer Agent competition without a GPU.

Applies the same changes as sections 2b and 6 of gemma-4-developer-agent-v2.ipynb
(prompt, budgets, sampling, adapter removal) to the competition's sample_submission
folder and zips it. No vLLM, no model, no local evaluation.

Kaggle (CPU notebook, Accelerator: None, competition data attached):
    !python build_submission.py
Local (after downloading sample_submission/ from the competition Data tab):
    python build_submission.py --src path/to/sample_submission --out ./out
"""
import argparse
import re
import shutil
import zipfile
from pathlib import Path

import yaml

KAGGLE_SRC = Path('/kaggle/input/competitions/gemma-4-developer-agent/sample_submission')
KAGGLE_OUT = Path('/kaggle/working')

# =====================================================================
# Settings you may want to tweak between experiments
# =====================================================================
REMOVE_PLACEHOLDER_ADAPTERS = False  # the baseline (which scored) keeps them; agent.yaml references them
MAX_TIME_MINUTES = 60                # value from the submission that succeeded; 5 coincided with Kaggle Errors
MAX_TOOL_CALLS = 100                 # baseline value
MAX_TURNS = 50                       # baseline value; higher may exceed the allowed limit

SAMPLING_YAML = (
    "temperature: 0.2\n"
    "top_p: 0.95\n"
    "max_output_tokens: 8192\n"
    "thinking_config:\n"
    "  thinking_budget: 2048\n"
    "  include_thoughts: false\n"
)

SYSTEM_PROMPT = """\
You are an expert Python software engineer. Fix the issue described in the task by editing
the repository in your current working directory, then call submit_patch. Hidden unit tests
will be run against your patch; the task counts as solved only if those tests pass.

## Workflow (follow in order)
1. UNDERSTAND: Read the issue and hints carefully. Note the exact expected behavior and every
   function/class/parameter/error name mentioned. Hidden tests use those exact names and
   behaviors, so for feature requests implement exactly what the issue describes.
2. LOCATE: Use run_command with `grep -rn "<identifier>" --include=*.py .` to find the code.
   Focus on the library package, not docs/ or examples (exception: for FastAPI
   documentation-code tasks, edit the executable code under docs_src/). search_similar_code and
   get_code_neighbors work best with real symbol names you have already found
   (e.g. `package.module.ClassName`), not with English phrases.
3. READ: read_file the relevant functions using line ranges. Read the existing tests for that
   module (e.g. `ls tests | grep <module>`) to learn conventions and expected error types.
4. REPRODUCE: Write a tiny script in /tmp (e.g. /tmp/repro.py, NEVER inside the repository)
   that demonstrates the bug, and run it with `python /tmp/repro.py`.
5. FIX: Make the smallest correct change in the source code. Follow the existing code style.
   Fix the general case, not just the example in the issue. Keep the public API unchanged
   unless the issue asks for a change. Use edit_file; old_string must match the file exactly,
   including indentation.
6. VERIFY: Re-run /tmp/repro.py. Then run the related existing tests:
   `python -m pytest tests/<relevant_test_file>.py -x -q 2>&1 | tail -30`.
   If anything fails because of your change, fix it. Existing tests may already be broken
   (missing fixtures, import errors) for reasons unrelated to your task: ignore those and
   never try to repair them.
7. CLEAN UP: Run `git status` and `git diff`. Delete any stray files you created inside the
   repository. Do NOT modify or add files under tests/, docs/, or changelog/release notes.
8. SUBMIT: Call submit_patch and check that patch_size > 0 and files_changed > 0.

## Rules
- Always call submit_patch before you run out of budget. An imperfect patch is better than
  none. Use get_status to check your budget; when about 70% is used, finish and submit.
- Every task needs a source change. Never conclude that nothing needs fixing.
- NEVER run bare `pytest`, `pytest .` or `python -m unittest discover`. Full test suites take
  minutes and exhaust your time budget. Always name a specific test file.
- The repository is in /workspace. Never search outside it (e.g. /usr/local/lib, /wheels,
  /opt). Dependencies are already installed.
- Do not explore aimlessly: if 3 searches find nothing, change strategy (broader grep, list
  the package directory, read the module __init__.py).
- Prefer targeted edit_file changes over rewriting whole files with write_file.
- Do not install packages; there is no internet.
- Keep command output short (pipe through head, tail or grep).
"""


def remove_placeholder_adapters(agent_dir: Path) -> None:
    if not (agent_dir / 'adapters').exists():
        return
    yaml_files = [agent_dir / 'agent.yaml', *(agent_dir / 'sub_agents').glob('*.yaml')]
    originals = {y: y.read_text(encoding='utf-8') for y in yaml_files}
    adapter_names = [d.name for d in (agent_dir / 'adapters').iterdir() if d.is_dir()]
    for y, txt in originals.items():
        y.write_text(re.sub(r'(?m)^[ \t]*adapter:[ \t]*\S+[ \t]*\n', '', txt), encoding='utf-8', newline='\n')
    still_referenced = [
        n for n in adapter_names
        if any(n in y.read_text(encoding='utf-8') for y in yaml_files)
    ]
    if still_referenced:
        for y, txt in originals.items():
            y.write_text(txt, encoding='utf-8', newline='\n')
        print(f'[!] Adapters still referenced elsewhere ({still_referenced}); kept them unchanged.')
    else:
        shutil.rmtree(agent_dir / 'adapters')
        print(f'Removed placeholder adapters: {adapter_names}')


def set_budgets(agent_dir: Path) -> None:
    cfg_path = agent_dir / 'eval_config.yaml'
    raw = yaml.safe_load(cfg_path.read_text(encoding='utf-8')) or {}
    section = raw['evaluation'] if 'evaluation' in raw else raw
    print('OLD eval_config:', dict(section))
    section['max_time_minutes'] = MAX_TIME_MINUTES
    section['max_tool_calls'] = MAX_TOOL_CALLS
    section['max_turns'] = MAX_TURNS
    cfg_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding='utf-8', newline='\n')
    print('NEW eval_config:', dict(section))


def build_zip(agent_dir: Path, out_dir: Path) -> Path:
    zip_path = Path(shutil.make_archive(str(out_dir / 'submission'), 'zip', root_dir=agent_dir))
    try:  # official limits, available where the competition wheelhouse is installed
        from swegemma.config import ALLOWED_SUBMISSION_EXTENSIONS, MAX_SUBMISSION_SIZE_BYTES
    except ImportError:
        ALLOWED_SUBMISSION_EXTENSIONS, MAX_SUBMISSION_SIZE_BYTES = None, 3 * 1024**3
        print('[i] swegemma not installed; skipping the file-extension check.')
    with zipfile.ZipFile(zip_path, 'r') as zf:
        infos = zf.infolist()
        names = [i.filename for i in infos]
        assert 'agent.yaml' in names, 'agent.yaml must be at the root of the zip'
        total_size = sum(i.file_size for i in infos)
        assert total_size <= MAX_SUBMISSION_SIZE_BYTES, f'Archive exceeds 3 GiB limit: {total_size}'
        if ALLOWED_SUBMISSION_EXTENSIONS is not None:
            for info in infos:
                if not info.is_dir():
                    ext = Path(info.filename).suffix.lower()
                    assert ext in ALLOWED_SUBMISSION_EXTENSIONS, f'Disallowed file extension: {info.filename}'
        print('Zip contents:')
        for n in names:
            print('  ', n)
    print(f'\nCreated {zip_path} ({zip_path.stat().st_size / (1024 * 1024):.2f} MiB)')
    return zip_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--src', type=Path, default=KAGGLE_SRC,
                        help='sample_submission folder or a previous submission.zip')
    parser.add_argument('--out', type=Path, default=KAGGLE_OUT if KAGGLE_OUT.exists() else Path('out'))
    args = parser.parse_args()

    if args.src.suffix.lower() == '.zip':  # a previous submission.zip works as the source
        extracted = args.out / '_src'
        if extracted.exists():
            shutil.rmtree(extracted)
        with zipfile.ZipFile(args.src) as zf:
            zf.extractall(extracted)
        roots = [p.parent for p in extracted.rglob('agent.yaml')]
        args.src = min(roots, key=lambda p: len(p.parts)) if roots else extracted

    if not (args.src / 'agent.yaml').exists():
        raise SystemExit(f'agent.yaml not found in {args.src}; pass --src <sample_submission folder>')
    args.out.mkdir(parents=True, exist_ok=True)
    agent_dir = args.out / 'sample_submission'
    if agent_dir.exists():
        shutil.rmtree(agent_dir)
    shutil.copytree(args.src, agent_dir)

    if REMOVE_PLACEHOLDER_ADAPTERS:
        remove_placeholder_adapters(agent_dir)
    set_budgets(agent_dir)
    (agent_dir / 'configs' / 'sampling.yaml').write_text(SAMPLING_YAML, encoding='utf-8', newline='\n')
    (agent_dir / 'prompts' / 'system.md').write_text(SYSTEM_PROMPT, encoding='utf-8', newline='\n')

    agent_yaml_text = (agent_dir / 'agent.yaml').read_text(encoding='utf-8')
    print(f'\n===== FINAL agent.yaml =====\n{agent_yaml_text}')
    if 'prompts/system.md' not in agent_yaml_text:
        print('[!] WARNING: agent.yaml does not reference prompts/system.md - the new prompt may not be used!')
    tool_names = ['run_command', 'read_file', 'edit_file', 'write_file',
                  'search_similar_code', 'get_code_neighbors', 'get_status', 'submit_patch']
    all_yaml = ''.join(p.read_text(encoding='utf-8') for p in agent_dir.rglob('*.yaml'))
    missing = [t for t in tool_names if t not in all_yaml]
    if missing:
        print(f'[!] Tools named in the prompt but not found in any YAML (check names): {missing}')

    build_zip(agent_dir, args.out)


if __name__ == '__main__':
    main()
