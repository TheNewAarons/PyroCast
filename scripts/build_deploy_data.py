"""Construye el paquete de datos de despliegue (serving/deploy_bundle.py) desde
`data/processed/` y actualiza `deploy/vercel/data-bundle.json` con su URL de
GitHub Release y su sha256. Ver DEPLOY.md, paso "Publicar los datos".

Uso:
    uv run --package serving python scripts/build_deploy_data.py --tag data-2026-10-01
"""
import argparse
import json
from pathlib import Path

from serving.deploy_bundle import build_bundle

REPO = "TheNewAarons/PyroCast"
ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--tag", required=True,
                        help="tag del GitHub Release, p. ej. data-2026-10-01")
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "dist")
    args = parser.parse_args()

    name = f"pyrocast-{args.tag}.tar.gz"
    info = build_bundle(args.processed, args.out_dir / name)
    manifest = {
        "url": f"https://github.com/{args.repo}/releases/download/{args.tag}/{name}",
        "sha256": info.sha256,
        "bytes": info.bytes,
        "tag": args.tag,
        "files": len(info.files),
    }
    target = ROOT / "deploy" / "vercel" / "data-bundle.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{info.path} ({info.bytes / 1e6:.1f} MB, {len(info.files)} archivos)")
    print(f"sha256 {info.sha256}")
    print(f"-> {target}")
    print(f"Publicar: gh release create {args.tag} {info.path} --repo {args.repo} "
          f"--title 'Datos de despliegue {args.tag}' --notes-file <(tar -xOzf {info.path} "
          f"processed/ATTRIBUTION.txt)")


if __name__ == "__main__":
    main()
