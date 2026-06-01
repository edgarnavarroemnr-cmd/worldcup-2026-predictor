# ⚽ World Cup 2026 Predictor & Insights

> **Bilingual sports-analytics project** — predicting match outcomes and uncovering historical insights from 96 years of FIFA World Cup data, ahead of the 2026 tournament hosted by Mexico, the USA & Canada.
>
> **Proyecto bilingüe de análisis deportivo** — predicción de resultados e insights históricos a partir de 96 años de datos del Mundial, rumbo a la Copa 2026 organizada por México, EE.UU. y Canadá.

**Author / Autor:** Edgar Navarro · *Bilingual Data Analyst | Python · SQL · Tableau*
**Stack:** Python (pandas, scikit-learn) · PostgreSQL · Data Visualization

---

## 🇬🇧 English

### The question
Can we predict the outcome of a World Cup match using only what was knowable *before kickoff* — and which factors matter most? With 2026 around the corner (and Mexico co-hosting), I built an end-to-end pipeline that answers both.

### What I built
A three-phase pipeline:

1. **Data & SQL** — Cleaned 960 men's World Cup matches (1930–2022) from the academically-sourced [Fjelstul World Cup Database](https://github.com/jfjelstul/worldcup) into a relational PostgreSQL schema. Wrote analytical queries using window functions and CTEs (`worldcup_postgres.sql`).
2. **Feature engineering & modeling** — Built **leakage-free** features (every variable uses only pre-match information: historical win rate, scoring average, recent form, head-to-head record). Trained and compared Logistic Regression vs Random Forest.
3. **Visualization** — An interactive bilingual dashboard (`worldcup_2026_dashboard.html`).

### Key results

| Model | Accuracy | Log loss | Brier | AUC |
|---|---|---|---|---|
| Logistic Regression | 68.8% | 0.671 | 0.228 | 0.674 |
| **Random Forest** ⭐ | **69.5%** | **0.631** | **0.217** | **0.712** |

> **Honest evaluation:** I used a *temporal* split — trained on World Cups through 2014, tested on tournaments the model had never seen (2018 & 2022). A 69.5% accuracy over a 43% baseline is realistic; a 95% result would have been a red flag for data leakage.

### What I learned about the data
- **Historical scoring average and tournament experience** are the strongest predictors of a result — pedigree matters.
- The World Cup became **markedly more defensive** after the 1950s (from ~5 goals/match to ~2.7).
- **Mexico** ranks 24th of 85 all-time — consistent qualifier, with its historic peak at *Mexico 1986* (2.2 points/game).

### Files
| File | Description |
|---|---|
| `01_build_database.py` | Data cleaning + leakage-free feature engineering |
| `02_model.py` | Model training, temporal validation, evaluation |
| `03_visualize.py` | Portfolio charts (matplotlib) |
| `worldcup_postgres.sql` | PostgreSQL schema + analytical queries |
| `worldcup_2026_dashboard.html` | Interactive bilingual dashboard |
| `output/` | Generated database, datasets, figures |

---

## 🇲🇽 Español

### La pregunta
¿Se puede predecir el resultado de un partido de Mundial usando solo lo que se sabía *antes del silbatazo inicial* — y qué factores pesan más? Con el 2026 a la vuelta de la esquina (y México como anfitrión), construí un pipeline completo que responde ambas.

### Qué construí
Un pipeline en tres fases:

1. **Datos y SQL** — Limpié 960 partidos masculinos del Mundial (1930–2022) de la [Fjelstul World Cup Database](https://github.com/jfjelstul/worldcup) en un esquema relacional de PostgreSQL. Escribí consultas analíticas con funciones de ventana y CTEs (`worldcup_postgres.sql`).
2. **Ingeniería de variables y modelado** — Creé variables **sin fuga de datos** (cada una usa solo información previa al partido: tasa de victorias histórica, promedio goleador, forma reciente, historial directo). Entrené y comparé Regresión Logística vs Random Forest.
3. **Visualización** — Un dashboard interactivo bilingüe (`worldcup_2026_dashboard.html`).

### Resultados clave

| Modelo | Precisión | Log loss | Brier | AUC |
|---|---|---|---|---|
| Regresión Logística | 68.8% | 0.671 | 0.228 | 0.674 |
| **Random Forest** ⭐ | **69.5%** | **0.631** | **0.217** | **0.712** |

> **Evaluación honesta:** Usé una división *temporal* — entrené con Mundiales hasta 2014 y probé con torneos que el modelo nunca vio (2018 y 2022). Un 69.5% de precisión sobre un baseline de 43% es realista; un 95% habría sido señal de fuga de datos.

### Lo que aprendí de los datos
- El **promedio goleador histórico y la experiencia** son los predictores más fuertes — la jerarquía importa.
- El Mundial se volvió **mucho más defensivo** después de los años 50 (de ~5 goles/partido a ~2.7).
- **México** ocupa el lugar 24 de 85 históricamente — clasificador constante, con su pico en *México 1986* (2.2 puntos/partido).

---

## How to run / Cómo ejecutar

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
python 01_build_database.py    # builds DB + features
python 02_model.py             # trains & evaluates models
python 03_visualize.py         # generates figures

# PostgreSQL queries:
createdb worldcup
psql worldcup -f worldcup_postgres.sql
```

Open `output/worldcup_2026_dashboard.html` in any browser for the interactive dashboard.

---

*Data: Fjelstul World Cup Database (CC-BY-SA 4.0). This project is for educational & portfolio purposes.*
