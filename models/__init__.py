from models.CaDM_S.cadm_s import build_cadm_s
from models.CaDM_L.cadm_l import build_cadm_l


def build_model(args):
    if args.model_name=="cadm_s":
        return build_cadm_s(args)
    else:
        return build_cadm_l(args)
