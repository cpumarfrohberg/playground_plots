# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 16:39:45 2024

@author: Richard
"""

#necessary libraries 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency


# Load the Excel file
PATH = '~/wb/alfatraining/stats/materials/data/projekt'
xls = pd.ExcelFile(f'{PATH}/1_Marketing.xlsx')


# Load data from the 'Aufgabe1_Kinder' sheet
df_kinder = xls.parse('Aufgabe1_Kinder')

# Create a contingency table for the chi-square test for each gender
contingency_boys = pd.crosstab(df_kinder[df_kinder['Geschlecht'] == 'Junge']['Alter'], df_kinder[df_kinder['Geschlecht'] == 'Junge']['Präferenz'])
contingency_girls = pd.crosstab(df_kinder[df_kinder['Geschlecht'] == 'Mädchen']['Alter'], df_kinder[df_kinder['Geschlecht'] == 'Mädchen']['Präferenz'])

# Perform the chi-square test for boys
chi2_boys, p_boys, dof_boys, expected_boys = chi2_contingency(contingency_boys)

# Perform the chi-square test for girls
chi2_girls, p_girls, dof_girls, expected_girls = chi2_contingency(contingency_girls)

# Results of the chi-square tests
chi2_test_results_boys = {
    'Chi2 Statistic': chi2_boys,
    'p-value': p_boys,
    'Degrees of Freedom': dof_boys,
    'Expected Frequencies': expected_boys
}

chi2_test_results_girls = {
    'Chi2 Statistic': chi2_girls,
    'p-value': p_girls,
    'Degrees of Freedom': dof_girls,
    'Expected Frequencies': expected_girls
}

chi2_test_results_boys, chi2_test_results_girls


# Ausgeben der Chi-Quadrat-Testergebnisse für Jungen und Mädchen auf die Konsole
print("Chi-Quadrat-Testergebnisse für Jungen:")
print(f"Chi-Quadrat-Statistik: {chi2_boys}")
print(f"p-Wert: {p_boys}")
print(f"Freiheitsgrade: {dof_boys}")
print("Erwartete Häufigkeiten:")
print(expected_boys)
print("\n")

print("Chi-Quadrat-Testergebnisse für Mädchen:")
print(f"Chi-Quadrat-Statistik: {chi2_girls}")
print(f"p-Wert: {p_girls}")
print(f"Freiheitsgrade: {dof_girls}")
print("Erwartete Häufigkeiten:")
print(expected_girls)


# Create contingency tables again for boys and girls
contingency_boys = pd.crosstab(df_kinder[df_kinder['Geschlecht'] == 'Junge']['Alter'], df_kinder[df_kinder['Geschlecht'] == 'Junge']['Präferenz'])
contingency_girls = pd.crosstab(df_kinder[df_kinder['Geschlecht'] == 'Mädchen']['Alter'], df_kinder[df_kinder['Geschlecht'] == 'Mädchen']['Präferenz'])

# Chi-square test for boys and girls to get expected frequencies
_, _, _, expected_boys = chi2_contingency(contingency_boys)
_, _, _, expected_girls = chi2_contingency(contingency_girls)

# Convert expected frequencies to DataFrame for easier plotting
expected_boys_df = pd.DataFrame(expected_boys, index=contingency_boys.index, columns=contingency_boys.columns)
expected_girls_df = pd.DataFrame(expected_girls, index=contingency_girls.index, columns=contingency_girls.columns)

# Plotting the observed vs expected frequencies for boys
fig, ax = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

contingency_boys.plot(kind='bar', ax=ax[0], color=['#1f77b4', '#ff7f0e'])
ax[0].set_title('Beobachtete Häufigkeiten für Jungen')
ax[0].set_ylabel('Häufigkeit')

expected_boys_df.plot(kind='bar', ax=ax[1], color=['#1f77b4', '#ff7f0e'], alpha=0.7)
ax[1].set_title('Erwartete Häufigkeiten für Jungen')
ax[1].set_ylabel('Häufigkeit')

plt.suptitle('Vergleich der beobachteten und erwarteten Häufigkeiten')

# Plotting the observed vs expected frequencies for girls
fig, ax = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

contingency_girls.plot(kind='bar', ax=ax[0], color=['#1f77b4', '#ff7f0e'])
ax[0].set_title('Beobachtete Häufigkeiten für Mädchen')
ax[0].set_ylabel('Häufigkeit')

expected_girls_df.plot(kind='bar', ax=ax[1], color=['#1f77b4', '#ff7f0e'], alpha=0.7)
ax[1].set_title('Erwartete Häufigkeiten für Mädchen')
ax[1].set_ylabel('Häufigkeit')

plt.suptitle('Vergleich der beobachteten und erwarteten Häufigkeiten für Mädchen')

plt.show()
