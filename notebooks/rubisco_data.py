#!/usr/bin/python

import pandas as pd
import numpy as np


def load_rubisco_data(fname):
	"""Loads the rubisco data.

	Args:
		fname: filename of csv to load data from.

	Returns:
		raw_df, kin_df where raw is all the data and kin_df is the filtered data.
	"""
	# Load the data from Excel, remove data points that are not comparable.
	raw_kin_df = pd.read_csv(fname, index_col=0)

	# Filtered DataFrame used for most plots below.
	# Only variants with all the data
	kin_df = raw_kin_df.copy()
	has_KMs = np.logical_and(np.isfinite(kin_df.KC), np.isfinite(kin_df.KO))
	has_kcats = np.logical_and(np.isfinite(kin_df.vO), np.isfinite(kin_df.vO))
	has_kons = np.logical_and(np.isfinite(kin_df.kon_C), np.isfinite(kin_df.kon_O))
	has_all = np.logical_and(np.logical_and(has_KMs, has_kcats), has_kons)
	kin_df = kin_df[has_all]

	return raw_kin_df, kin_df