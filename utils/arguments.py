import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Argument parser for Semantic Diffusion Model')
    parser.add_argument(
        '--seed', type=int, default=3407, help='Random seed'
    )
    parser.add_argument(
        '--prompt', type=str, default="A cute fox wearing a graudation gown.", help='Prompt for the model'
    )
    parser.add_argument(
        '--num_inference_steps', type=int, default=50, help='Number of inference steps'
    )
    parser.add_argument(
        '--num_intervention_steps', type=int, default=20, help='Number of intervention steps'
    )
    parser.add_argument(
        '--updating_rate', type=int, default=10, help='Updating rate for Semantic Diffusion'
    )
    parser.add_argument(
        '--attn_res', type=tuple, default=(16, 16), help='Resolution of attention'
    )
    parser.add_argument(
        '--semantic_refine', type=bool, default=True, help='Whether to use semantic refinement'
    )
    parser.add_argument(
        '--attention_disentanglement', type=bool, default=False, help='Whether to use attention disentanglement'
    )
    parser.add_argument(
        '--attribute_entangle_weight', type=float, default=1.0, help='Weight hyperparameter for attribute entanglement'
    )
    parser.add_argument(
        '--related_disentangle_weight', type=float, default=1.0, help='Weight hyperparameter for related disentanglement'
    )
    parser.add_argument(
        '--entropy_regularization_weight', type=float, default=0.6, help='Weight hyperparameter for entropy regularization'
    )
    parser.add_argument(
        '--inlcude_entities', type=bool, default=False, help='Whether to include entities'
    )
    parser.add_argument(
        '--pretrained_model', type=str, default='stabilityai/stable-diffusion-2-base', help='Pretrained model'
    )
    parser.add_argument(
        '--output_path', type=str, default='./output', help='Path to save the output image'
    )
    return parser.parse_args()