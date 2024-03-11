#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 10:59:23 2024

@author: surasaksangkhathat
"""

#Import basic packages from Colab
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from sklearn.preprocessing import OneHotEncoder
from scipy import stats
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sksurv.nonparametric import kaplan_meier_estimator
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

def soap_sheetin(path, sheetname = 'Sheet1'):
	data = pd.read_excel(path, sheet_name=sheetname, engine='openpyxl')
	return data

def soap_explore(data): #Display all column names, data types and dimension
	size = data.size
	dimension = data.shape
	variables = data.columns.values.tolist()
	print('===============================================================================================================')
	print('\nThe dataframe has',size, 'cells, with',dimension,'(row x column) dimension.\n')
	print('All the variables include;',variables,'\n')
	print('===============================================================================================================')
	print('Types and numbers of each variables are;')
	print(data.info())

def soap_describe(data):
	for var in data.columns.values.tolist():
		if data[var].nunique() > 2 and data[var].dtypes != 'O':
				print(var)
				des = (data[var].describe())
				print(des)
				plt.figure(figsize=(5, 2))
				sns.boxplot(x=data[var])
				plt.title(f'Boxplot of {var}')
				plt.xlabel(var)
				plt.show()
				print('-------------------------------------------------------------------------------------------------------------')

#Counting missing values in each column
def soap_unique_null(data):
    unique_values = {}
    null_count = data.isnull().sum().to_list()
    for col in data.columns:
        unique_values[col] = data[col].value_counts().shape[0]
    unique = pd.DataFrame(unique_values, index=['unique value count']).transpose()
    unique['number of null'] = null_count
    print(unique)

#Counting percent of categorical data
def soap_count_percent(data, col):
  abs_count = data[col].value_counts()
  rel_count = data[col].value_counts(normalize=True)*100
  count_tab = pd.DataFrame({'abs_count' : abs_count, 'percent' : rel_count})
  count_tab = count_tab.sort_index(ascending=True)
  count_tab.index.name = col
  print(count_tab)
  plt.figure(figsize=(5, 3))
  sns.barplot(x=col, y='percent', data=count_tab)
  plt.ylabel('Percentage')
  plt.title(f'Percentage of each value in {col}')
  plt.show()
  
#Counting percent of all categorical data
def soap_batch_percent(data):
  for i in data.columns:
    if data[i].nunique() < 6:
      print('------------------------------------------------------------------')
      soap_count_percent(data, i)
      
#Exploring if the data has normal distribution or not
def shapif(data):
    data_numeric = data.select_dtypes(exclude=['object'])

    d1 = {}
    for i in data_numeric.columns.values.tolist():
        data_numeric[i].astype('float64')
        x = data_numeric[i].dropna()
        s = shapiro(x)
        # Rounding each element of the tuple to 3 decimal places
        s_rounded = (round(s[0], 3), round(s[1], 3))
        d1[i] = s_rounded
    df1 = pd.DataFrame(d1.items(), columns=['variable', 'Shapiro-Wilk result'])
    print(df1)


def soaplore(data):
    num_col = []
    for i in data.columns:
      if data[i].nunique() > 5:
        num_col.append(i)
    data = data[num_col]
    shapif(data)
    print('\n')      

#Create one-hot encoder columns from a categorial column
def soap_onehot(data, col):
  encoder = OneHotEncoder(sparse=False)
  onehot_encoded_data = encoder.fit_transform(data[[col]])
  onehot_encoded_df = pd.DataFrame(onehot_encoded_data, columns=encoder.get_feature_names_out([col])).astype(int)
  result = pd.concat([data, onehot_encoded_df], axis=1)
  return result

#Change continuous variable to binary
def soap_genbi(data, var, cutoff):
  var_gr = []
  for i in data[var]:
    if i < cutoff:
      var_gr.append(0)
    if i >= cutoff:
      var_gr.append(1)
  data[f'{var}_{cutoff}'] = var_gr
  data.groupby(f'{var}_{cutoff}').count()

#Find correlation with a target binary
def soap_target_corr(data, target):
  data.drop(target, axis=1).corrwith(data[target]).plot(kind='bar', grid=True, figsize=(20, 8) , title="Correlation with"+target, color="Purple")

#Heatmap for binary column correlations
def soap_heatmap_corr(data):
  bi_cols = []
  for col in data.columns:
    if data[col].nunique() == 2:
      bi_cols.append(col)

  corr = data[bi_cols].corr()

  plt.figure(figsize=(20, 20))  # Adjust the figure size as needed
  sns.heatmap(corr,
              xticklabels=corr.columns,
              yticklabels=corr.columns,
              annot=True,      # Annotate each cell with the numeric value
              cmap='coolwarm', # Color map
              linewidths=.5)   # Line widths between cells

  plt.title('Heatmap of Correlation Among Attributes')
  plt.show()    

#Return a list of binary column names
def soap_bicol_list(data):
    bi_cols = []
    for col in data.columns:
      if data[col].nunique() == 2:
        bi_cols.append(col)
    return bi_cols

def chi_pv(data, outcome, factor):
  table = pd.crosstab(data[outcome], data[factor])
  c, p, dof, expected = chi2_contingency(table)
  return p

def soap_x_across(data, outcome):
    d = {}
    for factor in data.columns.values.tolist():
        if data[factor].nunique() > 5:
            continue
        elif factor == outcome:
            continue
        else:
            pv=chi_pv(data, outcome, factor)
            d[factor] = pv
    df = pd.DataFrame(d.items(), columns=['variable', 'Chisquare p-value'])
    print(df)









def n_percent(data, col):
  display = []
  abs_count = data[col].value_counts().sort_index()
  rel_count = data[col].value_counts(normalize=True).sort_index() * 100
  for i in range(data[col].nunique()):
    display_figure = f"{abs_count.iloc[i]} ({rel_count.iloc[i]:.2f}%)"
    display.append(display_figure)
  return display

#Chi-square test

def soap_x_tab(data, var_a, var_b):
    table = pd.crosstab(data[var_b], data[var_a])
    c, p, dof, expected = chi2_contingency(table)
    print('=================================================================================')
    print('Data dimension: ', table.shape)
    print(f'Chi-square value = {c:.4f}')
    print(f'Chi-square p-value = {p:.4f}')
    print('=================================================================================\n')
    twosub = data[[var_a, var_b]]
    var_b_list = data[var_b].unique().tolist()
    all_col_list = []
    for i in var_b_list:
        Bi = twosub[data[var_b] == i]
        Ci = Bi.groupby(var_a).count()
        dict = {Ci.columns[0]: f'{Ci.columns[0]}_{str(i)}'}
        Ci = Ci.rename(columns=dict)
        all_col_list.append(Ci)
    d = pd.concat(all_col_list, axis=1, join='outer').fillna(0)
    sum_row = d.aggregate('sum', axis=1)
    e = pd.concat([d, sum_row], axis=1, join='inner')
    e = e.rename(columns={0: 'horizonsum'})
    sum_col = e.aggregate('sum', axis=0)
    e = pd.concat([e, pd.DataFrame([sum_col], columns=e.columns)], ignore_index=True)
    lst_a = data[var_a].unique()
    lst_a.sort()
    lst_a = lst_a.tolist()
    lst_a.append('vertisum')
    e[var_a] = lst_a
    cols = e.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    e = e[cols]
    cols = list(e.columns)
    h = cols[1:-1]
    h.sort()
    cols[1:-1] = h
    e = e[cols]
    i = e.iloc[:, 1:].astype(int)
    e = e.iloc[:, :1]
    e = pd.concat([e, i], axis=1, join='inner')
    for i in cols:
        if i == 'horizonsum':
            continue
        if i == var_a:
            continue
        else:
            e['% ' + i] = ((e[i] / e['horizonsum'] * 100)).round(4).astype(float)
    print(e.to_string(index=False))
    if data[var_b].nunique() == 2:
        f = e.columns.tolist()
        var = e.iloc[0:-1][f[0]]
        outc = e.iloc[0:-1][f[-1]]
        fig = plt.figure(figsize=(3, 3))
        plt.bar(var, outc, label=var.name)
        plt.xlabel(var.name)
        plt.xticks(range(int(var.min()), int(var.max()) + 1))
        plt.ylabel(outc.name)
        plt.title(f'Chi-square p-value: {p: .5f}')
        plt.show()

def soap_tableone_cat(data, col, target):
  all = n_percent(data, col)
  val = np.sort(data[col].unique()).tolist()
  grouped = data.groupby(target)
  grouplist = []
  for group_df in grouped:
    grouplist.append(group_df)
  gr_0 = (grouplist[0])[1] #splited df with target = 0
  gr_1 = (grouplist[1])[1] #splited with target = 1

  if gr_0[col].nunique() == 2:
    no = n_percent(gr_0, col)
  elif gr_0[col].nunique() == 1:
    if gr_0[col].unique() == 0:
      no = n_percent(gr_0, col) + ['0 (0.00%)']
    elif gr_0[col].unique() == 1:
      no = ['0 (0.00%)'] + n_percent(gr_0, col)

  if gr_1[col].nunique() == 2:
    yes = n_percent(gr_1, col)
  elif gr_1[col].nunique() == 1:
    if gr_1[col].unique() == 0:
      yes = n_percent(gr_1, col) + ['0 (0.00%)']
    elif gr_1[col].unique() == 1:
      yes = ['0 (0.00%)'] + n_percent(gr_1, col)



  chi_p = chi_pv(data, target, col)
  chi = ['%.3f' %chi_p,'']

  if chi_p < 0.001:
    data = {'All':all, 'Val': val, target+'_neg': no, target+'_pos': yes, 'p-value': ['< 0.001','']}
  if chi_p >= 0.001:
    data = {'All':all, 'Val': val, target+'_neg': no, target+'_pos': yes, 'p-value': chi}
  table_0 = pd.DataFrame(data)
  table_0.index.name = ' '
  return table_0

#Make table one for binary data and a binary target
def soap_joined_tableone_cat(data, bicol_list, target):
  All = [f"{len(data)} (100.00%)"]
  Val = ['-']
  Name = ['All']
  grouped = data.groupby(target)
  grouplist = []
  for group_df in grouped:
    grouplist.append(group_df)
  gr_0 = (grouplist[0])[1]
  gr_1 = (grouplist[1])[1]

  No = [f"{len(gr_0)} ({len(gr_0)*100/len(data):.2f}%)"]
  Yes = [f"{len(gr_1)} ({len(gr_1)*100/len(data):.2f}%)"]
  Chi = ['-']
  if target in bicol_list:
    bicol_list.remove(target)

  for col in bicol_list:
    all = n_percent(data, col)
    val = np.sort(data[col].unique()).tolist()
    if gr_0[col].nunique() == 2:
      no = n_percent(gr_0, col)
    elif gr_0[col].nunique() == 1:
      if gr_0[col].unique() == 0:
        no = n_percent(gr_0, col) + ['0 (0.00%)']
      elif gr_0[col].unique() == 1:
        no = ['0 (0.00%)'] + n_percent(gr_0, col) #Corrected here already
    if gr_1[col].nunique() == 2:
      yes = n_percent(gr_1, col)
    elif gr_1[col].nunique() == 1:
      if gr_1[col].unique() == 0:
        yes = n_percent(gr_1, col) + ['0 (0.00%)']
      elif gr_1[col].unique() == 1:
        yes = ['0 (0.00%)'] + n_percent(gr_1, col)
    name = [f'{col}', '']
    chi_p = chi_pv(data, target, col)
    if chi_p < 0.001:
      chi = ['<0.001','']
    elif chi_p >= 0.001:
      chi = ['%.3f' %chi_p,'']

    All.extend(all)
    No.extend(no)
    Yes.extend(yes)
    Chi.extend(chi)
    Name.extend(name)
    Val.extend(val)
  data = {'Parameter':Name, 'Val': Val, 'All':All, f'{target}_neg': No, f'{target}_pos': Yes, 'p-value': Chi}
  table_1 = pd.DataFrame(data)
  table_1.index.name = ' '
  return table_1

#TTest and Mann-Whitney-U test
def soap_TU(data, col, target):
	if data[target].nunique() != 2:
		print(f'The outcome {target} is non-binary')
	else:
		var_by_outcome = data.groupby(target)[col].describe()
		print(col,'\n', var_by_outcome)
		cat1 = data[data[target] == 0]
		cat2 = data[data[target] == 1]
		print('-----------------------------------------------------------------\n')
		print(stats.ttest_ind(cat1[col].dropna(), cat2[col].dropna()))
		print(stats.mannwhitneyu(cat1[col].dropna(), cat2[col].dropna()))

		plt.boxplot([cat1[col].dropna(), cat2[col].dropna()], showmeans = True)
		plt.show()

def soap_meansd(data, col):
  ms = f'{data[col].mean():.2f}({data[col].std():.2f})'
  return ms

def soap_tableone_meansd(data, col, target):
  all = soap_meansd(data, col)
  grouped = data.groupby(target)
  grouplist = []
  for group_df in grouped:
    grouplist.append(group_df)
  gr_0 = (grouplist[0])[1] #Group with target = 0
  gr_1 = (grouplist[1])[1] #Group with target = 1
  no = soap_meansd(gr_0, col)
  yes = soap_meansd(gr_1, col)

  ttest_result = stats.ttest_ind(gr_0[col].dropna(), gr_1[col].dropna()) #ttest of col grouped by target binary
  t = ttest_result.pvalue

  if t < 0.001:
    data = {'All':all, target+'_neg': no, target+'_pos': yes, 'p-value': '< 0.001'}
  if t >= 0.001:
    data = {'All':all, target+'_neg': no, target+'_pos': yes, 'p-value': f'{t:.3f}'}

  table_0 = pd.DataFrame([data], index = [0])
  return table_0

def soap_numcol_list(data):
    num_cols = []
    for col in data.columns:
        if (data[col].dtype != 'object') & (data[col].nunique() > 5):
            num_cols.append(col)
    serial_words = ['serial', 'Serial', 'hn', 'HN', 'id', 'ID']
    num_cols = [item for item in num_cols if item not in serial_words]
    return num_cols

def soap_joined_tableone_meansd(data, numcol, target):
  Name = []
  All = []
  No = []
  Yes = []
  P = []
  for col in numcol:
    Name.append(col)
    All.append(soap_meansd(data, col))
    grouped = data.groupby(target)
    grouplist = []
    for group_df in grouped:
      grouplist.append(group_df)
    gr_0 = (grouplist[0])[1] #Group with target = 0
    gr_1 = (grouplist[1])[1] #Group with target = 1
    No.append(soap_meansd(gr_0, col))
    Yes.append(soap_meansd(gr_1, col))
    ttest_result = stats.ttest_ind(gr_0[col].dropna(), gr_1[col].dropna())
    t = ttest_result.pvalue
    if t < 0.001:
      P.append(f'<0.001')
    if t >= 0.001:
      P.append(f'{t:.3f}')
  data = {'Parameter':Name, 'All':All, f'{target}_neg': No, f'{target}_pos': Yes, 'p-value': P}
  table_1 = pd.DataFrame(data)
  table_1.index.name = ' '
  return table_1

def soap_medianiqr(data, col):
  mi = f'{data[col].median():.2f}({data[col].quantile(0.25):.2f}-{data[col].quantile(0.75):.2f})'
  return mi

def soap_tableone_medianiqr(data, col, target):
  all = soap_medianiqr(data, col)
  grouped = data.groupby(target)
  grouplist = []
  for group_df in grouped:
    grouplist.append(group_df)
  gr_0 = (grouplist[0])[1] #Group with target = 0
  gr_1 = (grouplist[1])[1] #Group with target = 1
  no = soap_medianiqr(gr_0, col)
  yes = soap_medianiqr(gr_1, col)

  utest_result = stats.mannwhitneyu(gr_0[col].dropna(), gr_1[col].dropna()) #Mann-Whitney-U test of col grouped by target binary
  u = utest_result.pvalue

  if u < 0.001:
    data = {'All':all, target+'_neg': no, target+'_pos': yes, 'p-value': '< 0.001'}
  if u >= 0.001:
    data = {'All':all, target+'_neg': no, target+'_pos': yes, 'p-value': f'{u:.3f}'}

  table_0 = pd.DataFrame([data], index = [0])
  return table_0

def soap_joined_tableone_medianiqr(data, numcol, target):
  Name = []
  All = []
  No = []
  Yes = []
  P = []
  for col in numcol:
    Name.append(col)
    All.append(soap_medianiqr(data, col))
    grouped = data.groupby(target)
    grouplist = []
    for group_df in grouped:
      grouplist.append(group_df)
    gr_0 = (grouplist[0])[1] #Group with target = 0
    gr_1 = (grouplist[1])[1] #Group with target = 1
    No.append(soap_medianiqr(gr_0, col))
    Yes.append(soap_medianiqr(gr_1, col))
    utest_result = stats.mannwhitneyu(gr_0[col].dropna(), gr_1[col].dropna()) #Mann-Whitney-U test of col grouped by target binary
    u = utest_result.pvalue
    if u < 0.001:
      P.append(f'<0.001')
    if u >= 0.001:
      P.append(f'{u:.3f}')
  data = {'Parameter':Name, 'All':All, f'{target}_neg': No, f'{target}_pos': Yes, 'p-value': P}
  table_1 = pd.DataFrame(data)
  table_1.index.name = ' '
  return table_1

def soap_autotableone(data, target):

  bicol_list = soap_bicol_list(data)

  All = [f"{len(data)} (100.00%)"]
  Val = ['-']
  Name = ['All']

  grouped = data.groupby(target)
  grouplist = []
  for group_df in grouped:
    grouplist.append(group_df)
  gr_0 = (grouplist[0])[1]
  gr_1 = (grouplist[1])[1]

  No = [f"{len(gr_0)} ({len(gr_0)*100/len(data):.2f}%)"]
  Yes = [f"{len(gr_1)} ({len(gr_1)*100/len(data):.2f}%)"]
  P = ['-']

  if target in bicol_list:
    bicol_list.remove(target)
  for col in bicol_list:
    all = n_percent(data, col)
    val = np.sort(data[col].unique()).tolist()
    if gr_0[col].nunique() == 2:
      no = n_percent(gr_0, col)
    elif gr_0[col].nunique() == 1:
      if gr_0[col].unique() == 0:
        no = n_percent(gr_0, col) + ['0 (0.00%)']
      elif gr_0[col].unique() == 1:
        no = ['0 (0.00%)'] + n_percent(gr_0, col) #Corrected here already
    if gr_1[col].nunique() == 2:
      yes = n_percent(gr_1, col)
    elif gr_1[col].nunique() == 1:
      if gr_1[col].unique() == 0:
        yes = n_percent(gr_1, col) + ['0 (0.00%)']
      elif gr_1[col].unique() == 1:
        yes = ['0 (0.00%)'] + n_percent(gr_1, col)
    name = [f'{col}', '']
    chi_p = chi_pv(data, target, col)
    if chi_p < 0.001:
      chi = ['<0.001','']
    elif chi_p >= 0.001:
      chi = ['%.3f' %chi_p,'']

    All.extend(all)
    No.extend(no)
    Yes.extend(yes)
    P.extend(chi)
    Name.extend(name)
    Val.extend(val)

    numcol = soap_numcol_list(data) #Extract numeric column excluding serial words
    normnumcol = [] #For normal distribution
    notnormnumcol = [] #For not_normal distribution
    for col in numcol:
      series_float64 = data[col].astype('float64').dropna()
      stat, p_value = shapiro(series_float64)
      if p_value >= 0.05:
        normnumcol.append(col)
      else:
        notnormnumcol.append(col)

  for col in notnormnumcol:
    Name.append(col)
    All.append(soap_medianiqr(data, col))
    Val.append('-')
    No.append(soap_medianiqr(gr_0, col))
    Yes.append(soap_medianiqr(gr_1, col))
    utest_result = stats.mannwhitneyu(gr_0[col].dropna(), gr_1[col].dropna()) #Mann-Whitney-U test of col grouped by target binary
    u = utest_result.pvalue
    if u < 0.001:
      P.append(f'<0.001')
    if u >= 0.001:
      P.append(f'{u:.3f}')

  for col in normnumcol:
    Name.append(col)
    All.append(soap_meansd(data, col))
    Val.append('-')
    No.append(soap_meansd(gr_0, col))
    Yes.append(soap_meansd(gr_1, col))
    ttest_result = stats.ttest_ind(gr_0[col].dropna(), gr_1[col].dropna())
    t = ttest_result.pvalue
    if t < 0.001:
      P.append(f'<0.001')
    if t >= 0.001:
      P.append(f'{t:.3f}')

  data = {'Parameter':Name, 'Val': Val, 'All':All, f'{target}_neg': No, f'{target}_pos': Yes, 'p-value': P}
  table_1 = pd.DataFrame(data)
  table_1.index.name = ' '
  return table_1

#Univariate logistic regression

def soap_unilogit(df, target, feature):
    df['intercept'] = 1.0
    X = df[[feature, 'intercept']]  # Independent variable(s)
    y = df[target]                  # Dependent variable
    model = sm.Logit(y, X).fit()
    params = model.params
    odds_ratios = np.exp(params)
    conf = model.conf_int()
    conf['Odds Ratio'] = odds_ratios
    conf.columns = ['2.5%', '97.5%', 'Odds Ratio']
    conf[['2.5%', '97.5%']] = np.exp(conf[['2.5%', '97.5%']])
    conf['p-value'] = model.pvalues
    conf = conf.applymap(lambda x: f'{x:.4f}')
    stat = conf[['Odds Ratio', '2.5%', '97.5%', 'p-value']]
    print(stat)

def soap_batch_unilogit(df, target, feature_list):
    le = LabelEncoder()
    df['intercept'] = 1.0
    Name = []
    OR = []
    CI = []
    P = []
    for feature in feature_list:
        if feature == target:
          continue
        elif df[feature].dtype == 'O':
            if df[feature].nunique() / len(df) < 0.05:
              df[feature] = le.fit_transform(df[feature])
        else:
          pass
        X = df[[feature, 'intercept']]  # Independent variable(s)
        y = df[target]                  # Dependent variable
        model = sm.Logit(y, X).fit(disp=0)
        params = model.params
        odds_ratios = np.exp(params)
        conf = model.conf_int()
        conf['Odds Ratio'] = odds_ratios
        conf.columns = ['2.5%', '97.5%', 'Odds Ratio']
        conf[['2.5%', '97.5%']] = np.exp(conf[['2.5%', '97.5%']])
        conf['p-value'] = model.pvalues

        # Keep the formatting to where it's needed and avoid global applymap
        OR_val = f'{odds_ratios[feature]:.4f}'
        CI_val = f"({conf.loc[feature, '2.5%']:.4f}-{conf.loc[feature, '97.5%']:.4f})"
        P_val = conf.loc[feature, 'p-value']

        if P_val < 0.001:
            P_val_formatted = "<0.001"
        else:
            P_val_formatted = f"{P_val:.4f}"

        Name.append(feature)
        OR.append(OR_val)
        CI.append(CI_val)
        P.append(P_val_formatted)

    data = {'Parameter': Name, 'Crude OR': OR, '95% CI': CI, 'p-value': P}
    table_1 = pd.DataFrame(data)
    table_1.index.name = ' '
    print(f'Univariate logistic regression analysis')
    print(table_1)

def soap_batch_unilogit_graph(df, target, feature_list):
    df['intercept'] = 1.0
    le = LabelEncoder()
    Lower = []
    Upper = []
    OR = []
    Name = []
    for feature in feature_list:
        if feature == target:
          continue
        elif df[feature].dtype == 'O':
            if df[feature].nunique() / len(df) < 0.05:
              df[feature] = le.fit_transform(df[feature])
        else:
          pass
        X = df[[feature, 'intercept']]  # Independent variable(s)
        y = df[target]                  # Dependent variable
        model = sm.Logit(y, X).fit(disp=0)
        params = model.params
        odds_ratios = np.exp(params)
        conf = model.conf_int()
        conf['Odds Ratio'] = odds_ratios
        conf.columns = ['2.5%', '97.5%', 'Odds Ratio']
        conf[['2.5%', '97.5%']] = np.exp(conf[['2.5%', '97.5%']])
        ood = odds_ratios[feature]
        low = conf.loc[feature, '2.5%']
        up = conf.loc[feature, '97.5%']

        Name.append(feature)
        Lower.append(low)
        Upper.append(up)
        OR.append(ood)
    data = {'var': Name, 'OR': OR, '2.5%': Lower, '97.5%': Upper}
    val = pd.DataFrame(data)
    val.set_index('var')
    plt.figure(figsize=(5, 10))  # Adjust the figure size as needed

    # Directly use 'var' DataFrame, ensure it's correctly defined and not None
    plt.errorbar(val['OR'], val['var'], xerr=[val['OR'] - val['2.5%'], val['97.5%'] - val['OR']], fmt='o', color='b', capsize=5, linestyle='None', marker='o')
    plt.axvline(x=1, linestyle='--', color='r', linewidth=1)
    plt.yticks(rotation=0)  # Adjust rotation if needed
    plt.xlabel('Odds Ratio (95% CI)')
    plt.ylabel('Variables')
    plt.xticks(rotation=90)  # Adjust for better label readability
    plt.title('Odds Ratios and 95% Confidence Intervals from Univariable Logistic Regression')
    plt.tight_layout()

    plt.show()
    
def soap_roc(data, var, target):
  X = np.array(data[var]).reshape(-1, 1)
  Y = data[target]
  model_lr = LogisticRegression()
  model_lr.fit(X,Y)
  false_positive_rate, true_positive_rate, threshold = roc_curve(Y, X)
  print('roc_auc_score for Logistic Regression: ', roc_auc_score(Y, X))

  plt.subplots(1, figsize=(10,10))
  plt.title('Receiver Operating Characteristic - Logistic regression')
  plt.plot(false_positive_rate, true_positive_rate)
  plt.plot([0, 1], ls="--")
  plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
  plt.ylabel('Sensitivity')
  plt.xlabel('1-Specificity')
  plt.show()

  roctab = roc_curve(Y, X)
  roctab
  roctable = pd.DataFrame({'Cutpoint': roctab[2], 'Specificity': 1-roctab[0], 'Sensitivity': roctab[1]})
  print(roctable)
  
def soap_multiple_lr(data, target):
  X = pd.get_dummies(data.drop(target, axis=1), drop_first=True)
  X.fillna(X.mean(), inplace=True)
  y = data[target]
  X = sm.add_constant(X)
  model = sm.Logit(y, X).fit(disp=0)
  print(model.summary())
  params = model.params
  conf = model.conf_int()
  conf['adjusted OR'] = params
  conf.columns = ['2.5%','97.5%', 'adjusted OR']
  np.set_printoptions(precision=4, suppress=True)
  val = np.exp(conf)
  val = val.applymap(lambda x: f'{x:.4f}')
  val['p-value'] = model.pvalues
  P = []
  for p in val['p-value']:
    if p < 0.001:
      formatted_p = "<0.001"
    else:
      formatted_p = f"{p:.4f}"
    P.append(formatted_p)
  val['p-value'] = P
  stat = val[['adjusted OR', '2.5%', '97.5%', 'p-value']]
  print(stat)

def soap_multiple_lr_hgraph(data, target):
  X = pd.get_dummies(data.drop(target, axis=1), drop_first=True)
  X.fillna(X.mean(), inplace=True)
  y = data[target]
  X = sm.add_constant(X)
  model = sm.Logit(y, X).fit(disp=0)
  params = model.params
  conf = model.conf_int()
  conf['adjusted OR'] = params
  conf.columns = ['2.5%','97.5%', 'adjusted OR']
  np.set_printoptions(precision=4, suppress=True)
  val = np.exp(conf)

  plt.figure(figsize=(15, 5))
  val = val[1:]
  plt.errorbar(val.index, val['adjusted OR'], yerr=[val['adjusted OR'] - val['2.5%'], val['97.5%'] - val['adjusted OR']], fmt='o', color='b', capsize=5)
  plt.axhline(y=1, linestyle='--', color='r', linewidth=1)
  plt.xticks(rotation=90)
  plt.xlabel('Variables')
  plt.ylabel('adjusted Odds Ratio (95% CI)')
  plt.yticks(rotation=90)
  plt.title('Adjusted Odds Ratios and 95% Confidence Intervals from Multivariable Logistic Regression')
  plt.tight_layout()

  plt.show()
  
def soap_multiple_lr_vgraph(data, target):
    X = pd.get_dummies(data.drop(target, axis=1), drop_first=True)
    X.fillna(X.mean(), inplace=True)
    y = data[target]
    X = sm.add_constant(X)
    model = sm.Logit(y, X).fit(disp=0)
    params = model.params
    conf = model.conf_int()
    conf['adjusted OR'] = params
    conf.columns = ['2.5%', '97.5%', 'adjusted OR']
    np.set_printoptions(precision=4, suppress=True)
    val = np.exp(conf)

    plt.figure(figsize=(5, 15))  # Adjust the figure size as needed
    val = val[1:]  # Exclude the intercept for plotting
    # Use 'plt.errorbar' with vertical orientation
    plt.errorbar(val['adjusted OR'], val.index, xerr=[val['adjusted OR'] - val['2.5%'], val['97.5%'] - val['adjusted OR']], fmt='o', color='b', capsize=5, linestyle='None', marker='o')
    plt.axvline(x=1, linestyle='--', color='r', linewidth=1)
    plt.yticks(rotation=0)  # Adjust rotation if needed
    plt.xlabel('Adjusted Odds Ratio (95% CI)')
    plt.ylabel('Variables')
    plt.xticks(rotation=90)  # Adjust for better label readability
    plt.title('Adjusted Odds Ratios and 95% Confidence Intervals from Multivariable Logistic Regression')
    plt.tight_layout()

    plt.show()
    
#Draw a Kaplan-Meier curve
def soap_single_kmc_(data, status, interval, max_time=None):
    sta = data[status].to_list()
    sta_bool = [bool(item) for item in sta]
    interv = data[interval]


    # If max_time is not provided, use the maximum value of interv
    if max_time is None:
        max_time = interv.max()

    kmf = KaplanMeierFitter()
    kmf.fit(durations=interv, event_observed=sta_bool)  # Fit the data into the model
    time, survival_prob = kmf.survival_function_.index, kmf.survival_function_["KM_estimate"]

    kmf.plot()
    plt.title('Kaplan-Meier Survival Curves')
    plt.ylabel("est. probability of survival")
    plt.xlabel("time ")

    plt.xlim(0, max_time)  # Use max_time to set the x-axis limit

    plt.show()

#Get survival ladder
def soap_survival_ladder(data, status, interval): #Get time-point specific survival probability with the closest 95%CI
  kmf = KaplanMeierFitter()
  interval = data[interval]
  event = data[status]
  kmf.fit(interval, event_observed=event)
  time_points = [30, 365, 730, 1095, 1460, 1825]
  survival_prob = kmf.predict(time_points)
  kmf_times = kmf.survival_function_.index.to_numpy()

  closest_time_points = np.searchsorted(kmf_times, time_points, side='left')
  closest_time_points = np.minimum(closest_time_points, len(kmf_times) - 1)
  closest_kmf_times = kmf_times[closest_time_points]
  closest_confidence_intervals = kmf.confidence_interval_.loc[closest_kmf_times]
  closest_confidence_intervals.reset_index(inplace=True)
  closest_confidence_intervals.rename(columns={'index': 'Closest_days'}, inplace=True)

  survival_prob_df = survival_prob.to_frame(name='Survival Probability')
  survival_prob_df.reset_index(inplace=True)
  survival_prob_df.rename(columns={'index': 'Days'}, inplace=True)
  survival_prob_df['95%CI lower'] = closest_confidence_intervals['KM_estimate_lower_0.95']
  survival_prob_df['95%CI upper'] = closest_confidence_intervals['KM_estimate_upper_0.95']
  print(survival_prob_df)
  
def soap_multigroup_kp(data, event, interval, intervention):
    kmf = KaplanMeierFitter()
    groups = data[intervention].unique()
    for group in groups:
        data_gr = data[data[intervention] == group]
        kmf.fit(data_gr[interval], event_observed=data_gr[event], label=group)
        kmf.plot()
    plt.title('Kaplan-Meier Survival Curves')
    plt.xlabel('Time')
    plt.ylabel('Survival Probability')
    plt.legend(title=intervention)
    plt.show()

def soap_logrank(data, event, interval, intervention):
    if data[intervention].nunique() != 2:
        print('As the factor', intervention, 'is non-binary, the Logrank statistics is not calculated.')
    else:
        interval_0 = []
        group_0 = data[data[intervention] == 0]
        for i in group_0[interval]:
            interval_0.append(i)
        T = interval_0

        interval_1 = []
        group_1 = data[data[intervention] == 1]
        for i in group_1[interval]:
            interval_1.append(i)
        T1 = interval_1

        censor_0 = []
        for j in group_0[event]:
            censor_0.append(j)
        E = censor_0

        censor_1 = []
        for k in group_1[event]:
            censor_1.append(k)
        E1 = censor_1
        results = logrank_test(T, T1, E, E1)
        if results.p_value >= 0.05:
            print('Log-rank for', intervention,' gives p-value at: %.5f' % results.p_value)
        else:
            print('Log-rank for', intervention,' gives p-value at: %.5f' % results.p_value, '**')

#Batch log-rank test
def soap_batchlogrank(data, event, interval, batch):
    for i in batch:
        soap_logrank(data, event, interval, i)

#Cox's proportional hazard analysis

def soap_single_chr(data, event, interval, intervention):
    from lifelines import CoxPHFitter
    markers = []
    markers.append(intervention)
    markers.extend([interval, event])
    data = data[markers] #Feature selection

    cph = CoxPHFitter()
    cph.fit(data, duration_col= interval, event_col=event)
    summary = cph.summary
    summary_table = summary[['exp(coef)','exp(coef) lower 95%', 'exp(coef) upper 95%']]
    summary_table = summary_table.rename(columns={'exp(coef)': 'Hazard Ratio', 'exp(coef) lower 95%': '95%CI lower', 'exp(coef) upper 95%': '95%CI upper'}, errors="raise")

    print(summary_table)

#Batch Cox's regression
def soap_batchcox(data, event, interval, batch):
    for i in batch:
        soap_single_chr(data, event, interval, i)
        
#Multivariable Cox's proportional hazard analysis
def soap_multi_chr(data, event, interval, intervention_list):
    from lifelines import CoxPHFitter
    markers = intervention_list
    markers.extend([interval, event])
    data = data[markers] #Feature selection

    cph = CoxPHFitter()
    cph.fit(data, duration_col= interval, event_col=event)
    summary = cph.summary
    summary_table = summary[['exp(coef)','exp(coef) lower 95%', 'exp(coef) upper 95%']]
    summary_table = summary_table.rename(columns={'exp(coef)': 'Hazard Ratio', 'exp(coef) lower 95%': '95%CI lower', 'exp(coef) upper 95%': '95%CI upper'}, errors="raise")

    print(summary_table)
















