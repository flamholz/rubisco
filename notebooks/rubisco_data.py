#!/usr/bin/python

import pandas as pd
import numpy as np


def load_rubisco_data():
	"""Convencience function to load the Rubisco data.

	Loads the merged and filtered dataset - this file should already be 
	filtered so that it contains no measurements of mutants etc.
	Note that we hardcode the name of the dataset file for convenience.

	Returns:
		raw_df, kin_df where raw is all the data and kin_df is the filtered data.
	"""
	# Load the data from Excel, remove data points that are not comparable.
	fname = '../data/DatasetS2_RubiscoKinetics_Merged.csv'
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


def filter_data(raw_kin_df):
	"""There are intentionally some duplicates in the raw data because it
	includes Savir data directly copied from some primary references.

	Args:
		raw_kin_df: the full dataset. 

	Returns:
		deduped, savir, nonsavir
	"""
	savir_pmid = '20142476'
	savir_df = raw_kin_df[raw_kin_df.pmid_or_doi == savir_pmid]
	nonsavir_df = raw_kin_df[raw_kin_df.pmid_or_doi != savir_pmid]

	measured_vars = 'KC, vC, S, KO'.split(', ')
	dup_rows = raw_kin_df.duplicated(subset=measured_vars, keep=False)
	dup_df = raw_kin_df[dup_rows]
	savir_dups = dup_df[dup_df.pmid_or_doi == savir_pmid]
	deduped = raw_kin_df.drop(savir_dups.index, axis=0)
	return deduped, savir_df, nonsavir_df


def merge_organisms(kin_df):
	"""Merge organisms by a median filter."""
	# Grouping by species and mutant just in case some of the mutations are not 
	# listed in the species name (they should all be, but mistakes happen). 

	kin_cols = 'KC, vC, S, KO, KRuBP, vO, kon_C, kon_O'.split(', ')
	metadata_cols = ['species', 'mutant', 'isoform', 'taxonomy', 'short_ref']
	cols2keep = kin_cols + metadata_cols
	subsset_cols_df = kin_df[cols2keep]
	# Note: median error may not correspond to median value, so we gotta ditch the error values. 
	return subsset_cols_df.groupby(['species', 'mutant', 'isoform', 'taxonomy']).median().reset_index()