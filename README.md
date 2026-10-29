# Prime

Experiments on the distribution of prime numbers and the gaps between them, with an interactive Streamlit dashboard.

## What's here

- **Sieve generator** (`Prime_code/Sieve.cpp`): a Sieve of Eratosthenes up to 2,000,000 that writes every prime and its gap from the previous prime to `primes_with_gaps.csv` (148,933 primes).
- **Notebook** (`Experiments/cringe_1.ipynb`): exploratory analysis of the CSV, including the Prime Number Theorem and the skewness of the gap distribution.
- **Dashboard** (`dashboard.py`, plus the later variants `dahboard2.py` and `cringe.py`): a Streamlit app with dark and light themes and pages for:
  - Overview
  - Prime Number Theorem
  - Prime density
  - Gap analysis
  - Ulam spiral
  - AKS primality test
  - ML: a linear regression model fitted to the prime data

## Running it

Generate the data (optional, the CSV is already included):

```bash
cd Prime_code
g++ -O2 Sieve.cpp -o Sieve
./Sieve
```

Start the dashboard from the repository root:

```bash
pip install streamlit pandas numpy scipy plotly scikit-learn
streamlit run dashboard.py
```

The dashboard reads `./Prime_code/primes_with_gaps.csv` by default.
