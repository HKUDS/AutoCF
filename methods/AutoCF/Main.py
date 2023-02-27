import torch as t
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from Params import args
from Model import Model, RandomMaskSubgraphs, LocalGraph
from DataHandler import DataHandler
import numpy as np
import pickle
from Utils.Utils import *
from Utils.Utils import contrast
import os
import setproctitle
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

class Coach:
	def __init__(self, handler):
		self.handler = handler

		print('USER', args.user, 'ITEM', args.item)
		print('NUM OF INTERACTIONS', self.handler.trnLoader.dataset.__len__())
		self.metrics = dict()
		mets = ['Loss', 'preLoss', 'Recall', 'NDCG']
		for met in mets:
			self.metrics['Train' + met] = list()
			self.metrics['Test' + met] = list()

	def makePrint(self, name, ep, reses, save):
		ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
		for metric in reses:
			val = reses[metric]
			ret += '%s = %.4f, ' % (metric, val)
			tem = name + metric
			if save and tem in self.metrics:
				self.metrics[tem].append(val)
		ret = ret[:-2] + '  '
		return ret

	def run(self):
		self.prepareModel()
		log('Model Prepared')
		if args.load_model != None:
			self.loadModel()
			stloc = len(self.metrics['TrainLoss']) * args.tstEpoch - (args.tstEpoch - 1)
		else:
			stloc = 0
			log('Model Initialized')
		bestRes = None
		for ep in range(stloc, args.epoch):
			tstFlag = (ep % args.tstEpoch == 0)
			reses = self.trainEpoch()
			log(self.makePrint('Train', ep, reses, tstFlag))
			if tstFlag:
				reses = self.testEpoch()
				log(self.makePrint('Test', ep, reses, tstFlag))
				self.saveHistory()
				bestRes = reses if bestRes is None or reses['Recall'] > bestRes['Recall'] else bestRes
			print()
		reses = self.testEpoch()
		log(self.makePrint('Test', args.epoch, reses, True))
		log(self.makePrint('Best Result', args.epoch, bestRes, True))
		self.saveHistory()

	def prepareModel(self):
		self.model = Model().cuda()
		self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)
		self.masker = RandomMaskSubgraphs()
		self.sampler = LocalGraph()
	
	def trainEpoch(self):
		trnLoader = self.handler.trnLoader
		trnLoader.dataset.negSampling()
		epLoss, epPreLoss = 0, 0
		steps = trnLoader.dataset.__len__() // args.batch
		for i, tem in enumerate(trnLoader):
			if i % args.fixSteps == 0:
				sampScores, seeds = self.sampler(self.handler.allOneAdj, self.model.getEgoEmbeds())
				encoderAdj, decoderAdj = self.masker(self.handler.torchBiAdj, seeds)
			ancs, poss, _ = tem
			ancs = ancs.long().cuda()
			poss = poss.long().cuda()
			usrEmbeds, itmEmbeds = self.model(encoderAdj, decoderAdj)
			ancEmbeds = usrEmbeds[ancs]
			posEmbeds = itmEmbeds[poss]
			
			bprLoss = (-t.sum(ancEmbeds * posEmbeds, dim=-1)).mean()
			regLoss = calcRegLoss(self.model) * args.reg

			contrastLoss = (contrast(ancs, usrEmbeds) + contrast(poss, itmEmbeds)) * args.ssl_reg + contrast(ancs, usrEmbeds, itmEmbeds)

			loss = bprLoss + regLoss + contrastLoss
			
			if i % args.fixSteps == 0:
				localGlobalLoss = -sampScores.mean()
				loss += localGlobalLoss
			epLoss += loss.item()
			epPreLoss += bprLoss.item()
			self.opt.zero_grad()
			loss.backward()
			self.opt.step()
			log('Step %d/%d: loss = %.1f, reg = %.1f, cl = %.1f   ' % (i, steps, loss, regLoss, contrastLoss), save=False, oneline=True)
		ret = dict()
		ret['Loss'] = epLoss / steps
		ret['preLoss'] = epPreLoss / steps
		return ret

	def testEpoch(self):
		tstLoader = self.handler.tstLoader
		epLoss, epRecall, epNdcg = [0] * 3
		i = 0
		num = tstLoader.dataset.__len__()
		steps = num // args.tstBat
		for usr, trnMask in tstLoader:
			i += 1
			usr = usr.long().cuda()
			trnMask = trnMask.cuda()
			usrEmbeds, itmEmbeds = self.model(self.handler.torchBiAdj, self.handler.torchBiAdj)

			allPreds = t.mm(usrEmbeds[usr], t.transpose(itmEmbeds, 1, 0)) * (1 - trnMask) - trnMask * 1e8
			_, topLocs = t.topk(allPreds, args.topk)
			recall, ndcg = self.calcRes(topLocs.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr)
			epRecall += recall
			epNdcg += ndcg
			log('Steps %d/%d: recall = %.1f, ndcg = %.1f          ' % (i, steps, recall, ndcg), save=False, oneline=True)
		ret = dict()
		ret['Recall'] = epRecall / num
		ret['NDCG'] = epNdcg / num
		return ret

	def calcRes(self, topLocs, tstLocs, batIds):
		assert topLocs.shape[0] == len(batIds)
		allRecall = allNdcg = 0
		for i in range(len(batIds)):
			temTopLocs = list(topLocs[i])
			temTstLocs = tstLocs[batIds[i]]
			tstNum = len(temTstLocs)
			maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, args.topk))])
			recall = dcg = 0
			for val in temTstLocs:
				if val in temTopLocs:
					recall += 1
					dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))
			recall = recall / tstNum
			ndcg = dcg / maxDcg
			allRecall += recall
			allNdcg += ndcg
		return allRecall, allNdcg

	def saveHistory(self):
		if args.epoch == 0:
			return
		with open('../../History/' + args.save_path + '.his', 'wb') as fs:
			pickle.dump(self.metrics, fs)

		content = {
			'model': self.model,
		}
		t.save(content, '../../Models/' + args.save_path + '.mod')
		log('Model Saved: %s' % args.save_path)

	def loadModel(self):
		ckp = t.load('../../Models/' + args.load_model + '.mod')
		self.model = ckp['model']
		self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)

		with open('../../History/' + args.load_model + '.his', 'rb') as fs:
			self.metrics = pickle.load(fs)
		log('Model Loaded')	

if __name__ == '__main__':
	logger.saveDefault = True
	
	log('Start')
	handler = DataHandler()
	handler.LoadData()
	log('Load Data')

	coach = Coach(handler)
	coach.run()