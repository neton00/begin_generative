r"""사용자가 직접 실행할 노트북 일괄 실행기.

  .venv\Scripts\python.exe run_assignments.py --week all
  .venv\Scripts\python.exe run_assignments.py --week 12 --device cpu
결과는 각 주차 outputs/*_executed.ipynb에 저장. 원본은 덮어쓰지 않음.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--week', choices=['12', '13', 'all'], default='all')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--timeout', type=int, default=3600)
    args = parser.parse_args()
    os.environ['GEN_DEVICE'] = args.device
    # 스크립트 실행에서 그래프 GUI로 막히지 않게 함. 인라인 그림은 결과 노트북에 저장.
    import nbformat
    from nbclient import NotebookClient

    root = Path(__file__).resolve().parent
    for week in (['12', '13'] if args.week == 'all' else [args.week]):
        paths = sorted((root / f'week{week}').glob('*.ipynb'))
        for path in paths:
            notebook = nbformat.read(path, as_version=4)
            client = NotebookClient(notebook, timeout=args.timeout, kernel_name='python3',
                                    resources={'metadata': {'path': str(path.parent)}})
            # python3 커널도 현재 실행 인터프리터를 사용하도록 지정.
            import sys
            client.km = __import__('jupyter_client').KernelManager(kernel_name='python3')
            client.km.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
            print(f'Running {path.name}', flush=True)
            out = path.parent / 'outputs' / f'{path.stem}_executed.ipynb'
            out.parent.mkdir(exist_ok=True)
            try:
                client.execute()
            finally:
                nbformat.write(notebook, out)
            print(f'Saved {out}', flush=True)


if __name__ == '__main__':
    main()
