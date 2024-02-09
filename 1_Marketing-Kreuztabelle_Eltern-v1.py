# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 18:05:03 2024

@author: Richard

Um eine Kreuztabelle mit Assoziationsmaßen zu erstellen, werden wir die Beziehung zwischen dem 
Geschlecht des Kindes und deren Kaufpräferenz analysieren. Dazu werden wir eine Kreuztabelle (
    Kontingenztafel) erstellen und ggf. Assoziationsmaße wie Chi-Quadrat-Statistik berechnen, 
um zu sehen, ob eine statistisch signifikante Beziehung zwischen dem Geschlecht des Kindes und 
der Kaufpräferenz besteht.

Dieses Programm wird die Kreuztabelle grafisch darstellen, wobei die verschiedenen Präferenzen 
in unterschiedlichen Farben für Jungen und Mädchen gestapelt werden. 
"""

#Erstellung der Kreuztabelle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency

# Laden der Excel-Datei und des spezifischen Reiters
# file_path = ('D:\\Weiterbildung\\KLR-343-Dozierende\\Projekt\\1_Marketing.xlsx')
sheet_name = 'Aufgabe1_Eltern'
PATH = '~/wb/alfatraining/stats/materials/data/projekt/1_Marketing.xlsx'

# Lesen der spezifischen Daten aus der Excel-Datei
df = pd.read_excel(PATH, sheet_name=sheet_name)

# Erstellung der Kreuztabelle
cross_tab = pd.crosstab(df['Kind'], df['Kaufpräferenz'])

# Berechnung des Chi-Quadrat-Tests
chi2, p, dof, expected = chi2_contingency(cross_tab)

# Berechnung des Phi-Koeffizienten (𝝋)
phi_coefficient = (chi2 / df.shape[0]) ** 0.5
# Bestimmung des Vorzeichens für Phi
sign = np.sign((cross_tab.iloc[0, 0] * cross_tab.iloc[1, 1]) - (cross_tab.iloc[0, 1] * cross_tab.iloc[1, 0]))
phi_coefficient = sign * phi_coefficient

# Berechnung von Yule's Q
ad_minus_bc = (cross_tab.iloc[0, 0] * cross_tab.iloc[1, 1]) - (cross_tab.iloc[0, 1] * cross_tab.iloc[1, 0])
ad_plus_bc = (cross_tab.iloc[0, 0] * cross_tab.iloc[1, 1]) + (cross_tab.iloc[0, 1] * cross_tab.iloc[1, 0])
yules_q = ad_minus_bc / ad_plus_bc

# Ausgabe der Assoziationsmaße
print(f"Phi-Koeffizient (𝝋): {phi_coefficient}")
print(f"Yule's Q: {yules_q}")

# Grafik der Kreuztabelle
cross_tab.plot(kind='bar', stacked=True)
plt.title('Kreuztabelle der Kaufpräferenzen')
plt.xlabel('Kind')
plt.ylabel('Anzahl')
plt.xticks(rotation=0)
plt.show()



