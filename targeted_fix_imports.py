from pathlib import Path
import json, sys

orig = Path(r"p:\\Reaserch Project\\DINOv2_Distillation_Colab.ipynb")
data = orig.read_bytes()
bak = orig.with_suffix(orig.suffix + ".importfix.bak")
bak.write_bytes(data)
print('backup', bak)

patterns = [b'import torch\n', b'import torch.nn as nn\n', b'import torch.optim\n']
new = data
for p in patterns:
    if p in new:
        new = new.replace(p, p.replace(b'\\n', b'\\\\n'))
        print('replaced', p)

out = orig.with_name(orig.stem + orig.suffix + '.importfixed' + orig.suffix)
out.write_bytes(new)
print('wrote', out)
try:
    json.loads(new.decode('utf-8'))
    print('json ok')
except Exception as e:
    print('json error', e)
    sys.exit(3)
