from src.semantic_diffusion_pipeline import SemanticDiffusionPipeline
from utils.arguments import parse_args
import torch


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    pipe = SemanticDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-base").to(device)
    img = pipe()['images'][0].save('./output.png')
