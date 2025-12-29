CREATE OR REPLACE VIEW v_latest_signal_per_symbol AS
SELECT DISTINCT ON (symbol, lookback) symbol, lookback, score_mode, as_of_date, score, signal, regime, thresholds, confidence
FROM prosperity_signals_daily
ORDER BY symbol, lookback, as_of_date DESC;

CREATE OR REPLACE VIEW v_latest_features_per_symbol AS
SELECT DISTINCT ON (symbol, lookback) symbol, lookback, as_of_date, mom_5, mom_21, mom_63, trend_sma20_50, volatility_ann, rsi14, volume_z20, last_close, regime
FROM prosperity_features_daily
ORDER BY symbol, lookback, as_of_date DESC;
