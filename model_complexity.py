"""Actual model parameter count and device-specific forward timing (not paper FPS)."""
import argparse
import json
import time
import torch
from train import load_config
from model import LAGRNet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--height', type=int, default=518)
    parser.add_argument('--width', type=int, default=518)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    cfg = load_config(args.config)
    kwargs = cfg.model_kwargs()
    kwargs['pretrained_backbone'] = False
    model = LAGRNet(**kwargs).to(args.device).eval()
    image = torch.randn(1, 3, args.height, args.width, device=args.device)
    times = []
    with torch.no_grad():
        for i in range(args.runs + 2):
            if args.device.startswith('cuda'):
                torch.cuda.synchronize()
            start = time.perf_counter()
            output = model(image)
            if args.device.startswith('cuda'):
                torch.cuda.synchronize()
            if i >= 2:
                times.append(time.perf_counter() - start)
    print(json.dumps({'parameters': sum(p.numel() for p in model.parameters()),
                      'inverse_depth_shape': list(output['inverse_depth'].shape),
                      'device': args.device, 'mean_seconds': sum(times)/len(times),
                      'note': 'Measured locally, not a reproduction of paper hardware or FLOPs.'}, indent=2))


if __name__ == '__main__':
    main()
