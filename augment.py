import argparse
import logging

from augmentation import CLIPRouter, LLMRewriter, run_llm_da


def main():
    parser = argparse.ArgumentParser(
        description='Offline LLM-DA: SPR/SFR rewriting with CLIP routing')
    parser.add_argument('--feature_path', required=True,
                        help='original feature pickle')
    parser.add_argument('--output_path', required=True,
                        help='augmented feature pickle to write')
    parser.add_argument('--frames_root', default=None,
                        help='per-sample frames: <id>.npz or <id>/*.jpg')
    parser.add_argument('--phi_cache', default=None,
                        help='JSON cache of CLIP scores')
    parser.add_argument('--tokenizer', default='bert-base-uncased',
                        help='bert-base-uncased (MOSI/MOSEI) or '
                             'bert-base-chinese (CH-SIMS)')
    parser.add_argument('--theta', type=float, default=0.25)
    parser.add_argument('--strategy', default='routed',
                        choices=['routed', 'spr', 'sfr'],
                        help="'routed' uses the CLIP gate; 'spr'/'sfr' force a "
                             "single strategy (ablation variants)")
    parser.add_argument('--clip_model', default='openai/clip-vit-base-patch32')
    parser.add_argument('--llm_model', default='gpt-3.5-turbo-0125')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dry_run', action='store_true')
    args = parser.parse_args()

    router = CLIPRouter(model_name=args.clip_model, theta=args.theta,
                        device=args.device)
    rewriter = LLMRewriter(model=args.llm_model, dry_run=args.dry_run)
    run_llm_da(args.feature_path, args.output_path,
               frames_root=args.frames_root, phi_cache=args.phi_cache,
               router=router, rewriter=rewriter,
               tokenizer_name=args.tokenizer, strategy=args.strategy)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
