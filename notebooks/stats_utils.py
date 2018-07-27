#!/usr/bin/python

import numpy as np


def combine_dists(means, stds, n=1000):
	"""Assumes normal distributions. Combines by equal-weighted bootstrapping.

	Args:
		means: an iterable of means.
		stds: an iterable of std deviations, same order as above.
		n: number of samples per distribution.

	Returns:
		mean, std of combined distribution.
	"""
	l = []
	for mean, std in zip(means, stds):
		l.append(np.random.normal(mean, std, n))

	combined = np.hstack(l)
	return np.mean(combined), np.std(combined)


class RubiscoKinetics(object):

	def __init__(self, vC, vC_SD, KC, KC_SD, KO, KO_SD, S, S_SD, 
				 vO=None, vO_95CI=None):
		self.vC = vC
		self.vC_SD = vC_SD
		self.KC = KC
		self.KC_SD = KC_SD
		self.KO = KO
		self.KO_SD = KO_SD
		self.S = S
		self.S_SD = S_SD

		self.vO = vO
		self.vO_95CI = vO_95CI

		self.kon_C = None
		self.kon_C_95CI = None

		self.kon_O = None
		self.kon_O_95CI = None

	def infer(self, n=1000):
		vC = np.random.normal(self.vC, self.vC_SD, n)
		KC = np.random.normal(self.KC, self.KC_SD, n)
		KO = np.random.normal(self.KO, self.KO_SD, n)
		S = np.random.normal(self.S, self.S_SD, n)

		# Since S = vC KO / (vO KC) then
		# vO = KO vC / (S KC)
		conf_range = [2.5, 97.5]

		vO_vals = KO * vC / (S * KC)
		self.vO = np.median(vO_vals)
		self.vO_95CI = np.percentile(vO_vals, conf_range)

		kon_C_vals = vC / KC

		# Since konO = vO/KO and vO = KO vC / (S KC) then
		# konO = vC / (S KC)
		kon_O_vals = vC / (S * KC)
		conf_range = [2.5, 97.5]

		self.kon_C = np.median(kon_C_vals)
		self.kon_C_95CI = np.percentile(kon_C_vals, conf_range)
		self.kon_O = np.median(kon_O_vals)
		self.kon_O_95CI = np.percentile(kon_O_vals, conf_range)

	@staticmethod
	def from_kv(kv):
		"""From some key-value store."""
		return RubiscoKinetics(kv['vC'], kv['vC_SD'], 
			kv['KC'], kv['KC_SD'], 
			kv['KO'], kv['KO_SD'], 
			kv['S'], kv['S_SD'])

