from cde.utils.io import load_time_series_csv
import numpy as np
import pandas as pd
from scipy.interpolate import InterpolatedUnivariateSpline


EURO_OIS_CSV = "../../data/2_Eurostoxx50/eur_ois.csv"
EUROSTOXX_CSV = "../../data/2_Eurostoxx50/eurostoxx50_prices_eod.csv"
EURO_TAIL_VARIATION_CSV = "../../data/2_Eurostoxx50/eurostoxx50_exp_tail_variation_measures.csv"
REALIZED_VOL_CSV = "../../data/2_Eurostoxx50/eurostoxx50_realized_volmeasures.csv"
RISKNEUTRAL_CSV = "../../data/2_Eurostoxx50/eurostoxx50_riskneutralmeasures.csv"
VRP_CSV = "../../data/2_Eurostoxx50/eurostoxx50_vrp.csv"
FAMA_FRENCH_CSV = "../../data/2_Eurostoxx50/FamaFrench_Europe_3_Factors_Daily.csv"
FAMA_FRENCH_MOMENTS_CSV = "../../data/2_Eurostoxx50/FamaFrench_Europe_MOM_Factor_Daily.csv"

def make_return_df(return_periods):
  eurostoxx = load_time_series_csv(EUROSTOXX_CSV)
  for h in return_periods:
    eurostoxx['log_ret_%i'%h] = np.log(eurostoxx.lastprice) - np.log(eurostoxx.lastprice.shift(h))

  # compute last period return
  eurostoxx['log_ret_last_period'] = (np.log(eurostoxx.lastprice) - np.log(eurostoxx.lastprice.shift(1))).shift(1)
  return eurostoxx.drop(labels=['lastprice'], axis=1)


def make_risk_free_df():
  euro_oid = load_time_series_csv(EURO_OIS_CSV)
  euro_oid = euro_oid[euro_oid.maturity == 1]
  euro_oid['log_risk_free_1d'] = np.log((euro_oid['yield']/365) + 1)
  return euro_oid.drop(labels=['maturity', 'yield'], axis=1)

def make_exp_tail_variation_df():
  return load_time_series_csv(EURO_TAIL_VARIATION_CSV)

def make_realized_vol_df():
  realized_vol = load_time_series_csv(REALIZED_VOL_CSV)
  return realized_vol.loc[:, ['RealizedVariation']]

def make_riskneutral_df(time_horizon):
  cols_of_interest = ['bakshiSkew', 'bakshiKurt', 'SVIX',]
  riskteural_measures = load_time_series_csv(RISKNEUTRAL_CSV, delimiter=';')
  riskteural_measures = riskteural_measures[['daystomaturity'] + cols_of_interest]
  interpolated_df = pd.DataFrame()
  for date in list(set(riskteural_measures.index)):
    # filter all row for respective date
    riskneutral_measures_per_day = riskteural_measures.ix[date]

    # filer out all option-implied measures with computed based on a maturity of less than 7 days
    riskneutral_measures_per_day = riskneutral_measures_per_day[riskneutral_measures_per_day['daystomaturity'] > 7]

    # interpolate / extrapolate to get estimate for desired time_horizon
    interpolated_values = [InterpolatedUnivariateSpline(np.array(riskneutral_measures_per_day['daystomaturity']),
                                 np.asarray(riskneutral_measures_per_day[col_of_interest]),
                                 k=1)(time_horizon) for col_of_interest in cols_of_interest]

    # create df with estimated option-implied risk measures
    update_dict = dict(zip(cols_of_interest, interpolated_values))
    update_dict.update({'daystomaturity': time_horizon})
    interpolated_df = interpolated_df.append(pd.DataFrame(update_dict, index=[date]))
  del interpolated_df['daystomaturity']
  return interpolated_df

def make_variance_risk_premium_df():
  return load_time_series_csv(VRP_CSV, delimiter=';')

def make_fama_french_df():
  fama_french_factors = load_time_series_csv(FAMA_FRENCH_CSV, time_format="%Y%m%d")
  return fama_french_factors.loc[:, ['Mkt-RF', 'SMB', 'HML']]

def make_fama_french_mom_df():
  return load_time_series_csv(FAMA_FRENCH_MOMENTS_CSV, time_format="%Y%m%d")

def compute_frama_french_factor_risk(df, time_steps):
  assert set(['WML', "Mkt-RF", 'SMB', 'HML']) <= set(df.columns)
  for ts in time_steps:
    for factor in ['WML', "Mkt-RF", 'SMB', 'HML']:
      df[factor + '_risk_%id'%ts] = df[factor].rolling(ts).sum()
  return df


def make_overall_eurostoxx_df(return_period=1):
  eurostoxx_returns = make_return_df(return_periods=[return_period])
  riskfree = make_risk_free_df()
  realized_vol = make_realized_vol_df()
  riskneutral_measures = make_riskneutral_df(time_horizon=30)
  fama_french = make_fama_french_df()
  fama_french_mom = make_fama_french_mom_df()

  df = eurostoxx_returns.join(riskfree, how='inner')
  df = df.join(realized_vol, how='inner')
  df = df.join(riskneutral_measures, how='inner')
  df = df.join(fama_french, how='inner')
  df = df.join(fama_french_mom, how='inner')  # add WML (winner-minus-looser) factor
  df = compute_frama_french_factor_risk(df, [10])
  return df


if __name__ == '__main__':
  df = make_overall_eurostoxx_df()
  print(df)