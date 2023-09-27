from from_scratch import ImageColorizerModel
import torch

def save_model(model: ImageColorizerModel, dir_path: str):
    torch.save(model.state_dict(), dir_path)

def load_model(dir_path: str):
    model = ImageColorizerModel()
    model.load_state_dict(torch.load(dir_path))
    model.eval()