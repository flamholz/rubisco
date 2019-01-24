#!/usr/bin/python

"""Functions for statistical inference of Rubisco parameter ranges.

The functions herein are mostly used for preprocessing the dataset 
in the "Normalize and Merge Raw Data" notebook. Since the specificity
S_C/O is related to the other 4 parameters, we can infer any one 
of the 5 commonly measured Rubisco kinetic parameters (kcat,C, KC,
kcat,O, KO, S_C/O) parameters from the other four. However, when we 
make these inferences we also want to report confidence intervals 
on the inference. We do this here with bootstrapping - sampling from
the distributions determined by the measurements and reported standard
deviations (assuming everything is normally distributed).

Note that for legacy reasons this code and the files produced use
the nomenclature of Savir et al. 2010 while the paper uses more
standard biochemical nomenclature. For this reason you see vC and
vO for kcat,C and kcat,O here. You will also see S = S_C/O since
it can be used as a variable name in python.
"""

__author__ = 'Avi Flamholz'


import numpy as np


def combine_dists(means, stds, n=1000):
	"""Assumes normal distributions. Combines by equal-weighted bootstrapping.

	Basic idea is that I have means and standard deviations of several 
	different measurements of the same quantity. I want to produce a 
	single mean and standard deviation for that quantity. So I assume 
	each of the distributions is normal with the given mean and standard
	deviation and I draw n (say 1000) samples from each. I then take the 
	union of the samples and report the mean and standard deviation of
	the union.

	This is used to merge multiple measurements of, say, S_C/O from the 
	same reference.

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
	"""Object that encapsulates kinetic data about single Rubisco.
	
	Encapsulates the work of calculating 95% CIs of derived parameters.
	It is generally the case that kcat,O (vO) is not measured directly
	therefore presumptively infer vO from the other parameters here.
	"""

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

	def has_carb(self):
		has_expr = [np.isfinite(self.vC),
					np.isfinite(self.vC_SD), 
					np.isfinite(self.KC),
					np.isfinite(self.KC_SD)]
		return np.all(has_expr)

	def has_all(self):
		has_expr = [np.isfinite(self.vC),
					np.isfinite(self.vC_SD), 
					np.isfinite(self.KC),
					np.isfinite(self.KC_SD),
					np.isfinite(self.KO),
					np.isfinite(self.KO_SD),
					np.isfinite(self.S),
					np.isfinite(self.S_SD)]
		return np.all(has_expr)

	def infer(self, n=1000):
		conf_range = [2.5, 97.5]

		if self.has_carb():
			vC = np.random.normal(self.vC, self.vC_SD, n)
			KC = np.random.normal(self.KC, self.KC_SD, n)

			kon_C_vals = vC / KC
			self.kon_C = np.median(kon_C_vals)
			self.kon_C_95CI = np.percentile(kon_C_vals, conf_range)

		if self.has_all():
			KO = np.random.normal(self.KO, self.KO_SD, n)
			S = np.random.normal(self.S, self.S_SD, n)

			# Since S = vC KO / (vO KC) then
			# vO = KO vC / (S KC)
			vO_vals = KO * vC / (S * KC)
			self.vO = np.median(vO_vals)
			self.vO_95CI = np.percentile(vO_vals, conf_range)

			# Since konO = vO/KO and vO = KO vC / (S KC) then
			# konO = vC / (S KC)
			kon_O_vals = vC / (S * KC)
			conf_range = [2.5, 97.5]

			self.kon_O = np.median(kon_O_vals)
			self.kon_O_95CI = np.percentile(kon_O_vals, conf_range)

	@staticmethod
	def from_kv(kv):
		"""From some key-value store."""
		return RubiscoKinetics(kv['vC'], kv['vC_SD'], 
			kv['KC'], kv['KC_SD'], 
			kv['KO'], kv['KO_SD'], 
			kv['S'], kv['S_SD'])

