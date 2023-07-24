
from .dcair import DCAIR
from .SASRec_model import SASRec
MODELS = {
    DCAIR.code(): DCAIR,
}

def model_factory(args):
    model = MODELS[args.model_code]
    if args.model_code =='dcair':
        return model(SASRec,args)
    else:
        return model(args)

