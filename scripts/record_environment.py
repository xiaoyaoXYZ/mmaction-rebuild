"""Record the isolated runtime and compare original package metadata inventory."""
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    import cv2
    import mmcv
    import mmengine
    import mmaction
    import numpy
    import psutil
    import torch
    import torchvision

    assert sys.prefix != sys.base_prefix, 'Expected an isolated venv'
    assert torch.cuda.is_available(), 'Expected working CUDA'
    old = json.loads((ROOT / 'reports/host_inventory_before.json').read_text(encoding='utf8'))
    changes = []
    for env in old['environments']:
        site = Path(env['path']) / 'Lib/site-packages'
        now = sorted([{'name': d.metadata['Name'], 'version': d.version}
                      for d in metadata.distributions(path=[str(site)])], key=lambda x: x['name'].lower())
        if now != env['packages']:
            changes.append({'path': env['path'], 'before': env['packages'], 'after': now})
    proc = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True)
    (ROOT / 'logs/pip_check.log').write_text(proc.stdout + proc.stderr, encoding='utf8')
    freeze = subprocess.run([sys.executable, '-m', 'pip', 'freeze', '--all'],
                            capture_output=True, text=True, check=True).stdout
    (ROOT / 'requirements-lock.txt').write_text(freeze, encoding='utf8')
    report = dict(
        python=sys.version, executable=sys.executable, prefix=sys.prefix,
        sys_path=sys.path,
        base_prefix=sys.base_prefix, venv_config=(Path(sys.prefix)/'pyvenv.cfg').read_text(),
        os=platform.platform(), processor=platform.processor(),
        logical_cpus=os.cpu_count(), memory_total_bytes=psutil.virtual_memory().total,
        gpu_name=torch.cuda.get_device_name(0), gpu_capability=list(torch.cuda.get_device_capability(0)),
        cuda_runtime=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
        torch_arch_list=torch.cuda.get_arch_list(),
        versions=dict(torch=torch.__version__, torchvision=torchvision.__version__,
                      mmcv_lite=mmcv.__version__, mmengine=mmengine.__version__,
                      mmaction2=mmaction.__version__, numpy=numpy.__version__, opencv=cv2.__version__),
        nvidia_smi=subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout,
        original_environments_package_metadata_unchanged=not changes,
        original_environment_changes=changes,
        pip_check_exit_code=proc.returncode, pip_check=proc.stdout.strip(),
        packages=sorted([{'name': d.metadata['Name'], 'version': d.version}
                         for d in metadata.distributions()], key=lambda x: x['name'].lower()))
    # A real CUDA kernel proves more than device enumeration.
    x = torch.tensor([1., 2., 3.], device='cuda', requires_grad=True)
    loss = (x * x).sum()
    loss.backward()
    torch.cuda.synchronize()
    report['cuda_kernel_probe'] = {'loss': loss.item(), 'gradient': x.grad.cpu().tolist()}
    assert report['cuda_kernel_probe'] == {'loss': 14.0, 'gradient': [2.0, 4.0, 6.0]}
    (ROOT / 'reports/environment_final.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf8')
    assert proc.returncode == 0, 'pip check failed; see logs/pip_check.log'
    assert not changes, 'Original package inventory changed; inspect report'
    print(json.dumps({k: report[k] for k in ['versions', 'gpu_name', 'gpu_capability',
                     'cuda_runtime', 'original_environments_package_metadata_unchanged',
                     'pip_check', 'cuda_kernel_probe']}, indent=2))


if __name__ == '__main__':
    main()
