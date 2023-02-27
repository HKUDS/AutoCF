import torch as t
import torch.nn.functional as F

def innerProduct(usrEmbeds, itmEmbeds):
	return t.sum(usrEmbeds * itmEmbeds, dim=-1)

def pairPredict(ancEmbeds, posEmbeds, negEmbeds):
	return innerProduct(ancEmbeds, posEmbeds) - innerProduct(ancEmbeds, negEmbeds)

def calcRegLoss(model):
	ret = 0
	for W in model.parameters():
		ret += W.norm(2).square()
	return ret

def contrast(nodes, allEmbeds, allEmbeds2=None):
	if allEmbeds2 is not None:
		pckEmbeds = allEmbeds[nodes]
		scores = t.log(t.exp(pckEmbeds @ allEmbeds2.T).sum(-1)).mean()
	else:
		uniqNodes = t.unique(nodes)
		pckEmbeds = allEmbeds[uniqNodes]
		scores = t.log(t.exp(pckEmbeds @ allEmbeds.T).sum(-1)).mean()
	return scores