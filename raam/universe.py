"""7Twelve universe per Giordano (2018), Table 1."""

from collections import OrderedDict

UNIVERSE = OrderedDict([
    ("VV",   ("US Equities",          "Vanguard Large-Cap ETF")),
    ("IJH",  ("US Equities",          "iShares Core S&P Mid-Cap ETF")),
    ("IJR",  ("US Equities",          "iShares Core S&P Small-Cap ETF")),
    ("EFA",  ("Intl Equities",        "iShares MSCI EAFE ETF")),
    ("EEM",  ("Intl Equities",        "iShares MSCI Emerging Markets ETF")),
    ("RWR",  ("Real Estate",          "SPDR Dow Jones REIT ETF")),
    ("DBC",  ("Resources/Commodities","Invesco DB Commodity Tracking ETF")),
    ("VAW",  ("Resources/Commodities","Vanguard Materials ETF")),
    ("AGG",  ("US Bonds",             "iShares Core US Aggregate Bond ETF")),
    ("TIP",  ("US Bonds",             "iShares TIPS Bond ETF")),
    ("IGOV", ("Intl Bonds",           "iShares International Treasury Bond ETF")),
    ("SHY",  ("Cash",                 "iShares 1-3 Year Treasury Bond ETF")),
])

CASH_TICKER = "SHY"
RANKED_TICKERS = [t for t in UNIVERSE if t != CASH_TICKER]
ASSET_CLASS = {t: meta[0] for t, meta in UNIVERSE.items()}
NAMES = {t: meta[1] for t, meta in UNIVERSE.items()}
