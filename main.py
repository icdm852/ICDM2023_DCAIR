
from options import args
from models.__init__ import model_factory
from dataloaders.__init__ import dataloader_factory
from trainers.__init__ import trainer_factory
from utils import *

def train():
    
    print(args.device)
    export_root = setup_train(args)
    
    train_loader, val_loader, test_loader= dataloader_factory(args)
    print('args.num_users+1:',args.num_users+1)
    print('args.num_items+1:',args.num_items+1)

    model = model_factory(args) 

    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, export_root)
    trainer.train()


    model.iteration=0 
    trainer.test(code='overall')
    model.iteration=0 
    trainer.test(code='head')
    model.iteration=0 
    trainer.test(code='tail')

def test():
    export_root = setup_train(args)
    train_loader, val_loader, test_loader= dataloader_factory(args)
    model = model_factory(args) 
 
    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, export_root)

    trainer.test(code='overall')
    trainer.test(code='head')
    trainer.test(code='tail')


if __name__ == '__main__':
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test()
    else:
        raise ValueError('Invalid mode')
