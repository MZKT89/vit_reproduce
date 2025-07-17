from models.vit import VisionTransformer
from data.dataset import get_dataloaders
from trainer.trainer import Trainer

def main():
    model = VisionTransformer()
    
    train_loader, val_loader = get_dataloaders()
    
    trainer = Trainer(model, train_loader, val_loader)
    trainer.train()


if __name__ == "__main__":
    main()