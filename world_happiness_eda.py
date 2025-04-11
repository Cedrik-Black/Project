
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Ustawienia wykresów
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# 2. Wczytanie danych
url = "https://raw.githubusercontent.com/iamspruce/World-Happiness-Report-2023/main/WHR2023.csv"
df = pd.read_csv(url)

# 3. Podstawowa analiza danych
print("👀 Podgląd danych:")
print(df.head())

print("\n📊 Informacje ogólne:")
print(df.info())

print("\n📈 Statystyki opisowe:")
print(df.describe())

# 4. Sprawdzenie braków danych
print("\n🔍 Braki danych:")
print(df.isnull().sum())

# 5. Korelacje
corr = df.corr(numeric_only=True)
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("📌 Korelacja między zmiennymi")
plt.show()

# 6. TOP 10 najszczęśliwszych krajów
top10 = df.sort_values(by="Ladder score", ascending=False).head(10)
sns.barplot(x="Ladder score", y="Country name", data=top10, palette="viridis")
plt.title("😊 TOP 10 najszczęśliwszych krajów (2023)")
plt.xlabel("Wynik szczęścia")
plt.ylabel("Kraj")
plt.show()

# 7. Zależność: PKB vs. Szczęście
sns.scatterplot(data=df, x="Logged GDP per capita", y="Ladder score", hue="Regional indicator", palette="tab10")
plt.title("💰 PKB vs. Poziom szczęścia")
plt.xlabel("Zalogowany PKB per capita")
plt.ylabel("Wynik szczęścia")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

# 8. Wpływ wolności i wsparcia społecznego
sns.lmplot(data=df, x="Freedom to make life choices", y="Ladder score", hue="Regional indicator", height=6, aspect=1.5)
plt.title("🕊️ Wolność wyboru a poziom szczęścia")
plt.show()

sns.lmplot(data=df, x="Social support", y="Ladder score", hue="Regional indicator", height=6, aspect=1.5)
plt.title("👥 Wsparcie społeczne a poziom szczęścia")
plt.show()
