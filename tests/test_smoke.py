from pathlib import Path
import subprocess

def test_cli_smoke(tmp_path: Path):
    img = Path('samples/house1.jpg')
    if not img.exists():
        return
    out = tmp_path / 'out'
    out.mkdir(exist_ok=True)
    cp = subprocess.run(['python', 'analyze.py', str(img), '--outdir', str(out)], capture_output=True)
    assert cp.returncode == 0
    svg = out / f"{img.stem}.svg"
    js = out / f"{img.stem}.json"
    assert svg.exists() and js.exists()