-- Migration 049: Regime Analog Library
-- Historical regime transitions with forward-return attribution by sector.
-- Seeded with 20+ events spanning 2000–2025.

CREATE TABLE IF NOT EXISTS regime_analog_library (
    analog_id                        TEXT        NOT NULL PRIMARY KEY,
    reference_date                   DATE        NOT NULL,
    regime_label                     TEXT        NOT NULL,
    macro_context                    JSONB       NOT NULL DEFAULT '{}'::jsonb,
    following_30d_return_by_sector   JSONB       NOT NULL DEFAULT '{}'::jsonb,
    following_90d_return_by_sector   JSONB       NOT NULL DEFAULT '{}'::jsonb,
    vix_at_entry                     NUMERIC,
    cape_at_entry                    NUMERIC,
    ic_state_at_entry                TEXT,
    description                      TEXT,
    created_at                       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_regime_analog_regime
    ON regime_analog_library (regime_label, reference_date DESC);

-- Seed: 20 historical regime transition events (2000–2025)
INSERT INTO regime_analog_library
    (analog_id, reference_date, regime_label, macro_context,
     following_30d_return_by_sector, following_90d_return_by_sector,
     vix_at_entry, cape_at_entry, ic_state_at_entry, description)
VALUES
-- 2000 dot-com peak → liquidity_fracture
('analog_2000_dotcom_peak',      '2000-03-10', 'liquidity_fracture',
 '{"rate_env":"rising","credit":"tightening","growth":"peak"}',
 '{"Technology":-15.2,"Financials":-4.1,"Healthcare":2.3,"Energy":1.8,"Utilities":3.5}',
 '{"Technology":-38.0,"Financials":-12.0,"Healthcare":5.0,"Energy":4.5,"Utilities":8.0}',
 33.0, 44.2, 'WEAK', 'Dot-com peak: extreme NASDAQ valuations, rate hikes'),

-- 2001 post-9/11 → transition_defensive
('analog_2001_911',              '2001-09-21', 'transition_defensive',
 '{"rate_env":"cutting","credit":"widening","growth":"recession","event":"geopolitical_shock"}',
 '{"Technology":-8.0,"Financials":-6.5,"Healthcare":4.0,"Energy":-3.0,"Utilities":5.2,"Defense":12.0}',
 '{"Technology":-5.0,"Financials":2.0,"Healthcare":8.0,"Energy":2.0,"Utilities":7.0,"Defense":18.0}',
 49.35, 32.0, 'INSUFFICIENT', 'Post-9/11 equity crash, flight to safety'),

-- 2003 recovery → recovery_selective
('analog_2003_recovery',         '2003-03-11', 'recovery_selective',
 '{"rate_env":"low","credit":"tightening","growth":"recovering"}',
 '{"Technology":15.5,"Financials":8.2,"Healthcare":4.0,"Energy":6.0,"Industrials":10.0}',
 '{"Technology":35.0,"Financials":22.0,"Healthcare":8.0,"Energy":20.0,"Industrials":25.0}',
 30.17, 25.0, 'MODERATE', 'Post-Iraq war recovery: value and cyclicals led'),

-- 2007 credit bubble → liquidity_fracture
('analog_2007_credit_peak',      '2007-07-19', 'liquidity_fracture',
 '{"rate_env":"peak","credit":"peak_tightness","growth":"decelerating","housing":"declining"}',
 '{"Technology":-2.0,"Financials":-12.0,"Healthcare":1.0,"Energy":5.0,"Utilities":2.0}',
 '{"Technology":-10.0,"Financials":-32.0,"Healthcare":-3.0,"Energy":10.0,"Utilities":-2.0}',
 16.12, 27.0, 'WEAK', 'Pre-GFC: credit markets cracking, housing in distress'),

-- 2008 Lehman collapse → transition_defensive
('analog_2008_lehman',           '2008-09-15', 'transition_defensive',
 '{"rate_env":"cutting","credit":"extreme_stress","growth":"contraction","event":"financial_crisis"}',
 '{"Technology":-22.0,"Financials":-35.0,"Healthcare":-10.0,"Energy":-25.0,"Utilities":-8.0}',
 '{"Technology":-28.0,"Financials":-40.0,"Healthcare":-15.0,"Energy":-35.0,"Utilities":-12.0}',
 79.13, 19.0, 'DEGRADED', 'GFC: systemic bank failure, credit freeze'),

-- 2009 trough → recovery_selective
('analog_2009_recovery',         '2009-03-09', 'recovery_selective',
 '{"rate_env":"zero","credit":"easing","growth":"trough","fiscal":"stimulus"}',
 '{"Technology":25.0,"Financials":65.0,"Healthcare":10.0,"Energy":22.0,"Industrials":35.0}',
 '{"Technology":60.0,"Financials":150.0,"Healthcare":25.0,"Energy":50.0,"Industrials":80.0}',
 52.65, 15.0, 'INSUFFICIENT', 'GFC trough: extreme value in financials and cyclicals'),

-- 2010 flash crash → transition_defensive
('analog_2010_flash_crash',      '2010-05-06', 'transition_defensive',
 '{"rate_env":"zero","credit":"stable","growth":"recovering","event":"flash_crash"}',
 '{"Technology":8.0,"Financials":5.0,"Healthcare":4.0,"Energy":6.0,"Utilities":3.0}',
 '{"Technology":18.0,"Financials":12.0,"Healthcare":10.0,"Energy":15.0,"Utilities":8.0}',
 45.79, 20.0, 'MODERATE', 'Flash crash: liquidity void, rapid recovery'),

-- 2011 European debt crisis → liquidity_fracture
('analog_2011_euro_crisis',      '2011-08-08', 'liquidity_fracture',
 '{"rate_env":"zero","credit":"euro_stress","growth":"slowing","event":"sovereign_crisis"}',
 '{"Technology":-5.0,"Financials":-18.0,"Healthcare":2.0,"Energy":-8.0,"Utilities":3.0}',
 '{"Technology":5.0,"Financials":-5.0,"Healthcare":8.0,"Energy":2.0,"Utilities":6.0}',
 48.0, 20.5, 'WEAK', 'European sovereign debt contagion, Greek/Italian spreads'),

-- 2013 Taper Tantrum → transition_defensive
('analog_2013_taper_tantrum',    '2013-06-19', 'transition_defensive',
 '{"rate_env":"rising_expectations","credit":"spread_widening","growth":"ok"}',
 '{"Technology":-2.0,"Financials":2.0,"Healthcare":1.0,"Energy":-3.0,"Utilities":-8.0}',
 '{"Technology":10.0,"Financials":8.0,"Healthcare":12.0,"Energy":3.0,"Utilities":-2.0}',
 20.49, 23.0, 'MODERATE', 'Fed taper signal: rate-sensitive assets sold'),

-- 2015-16 China slowdown → liquidity_fracture
('analog_2015_china',            '2015-08-24', 'liquidity_fracture',
 '{"rate_env":"first_hike","credit":"em_stress","growth":"china_slowdown"}',
 '{"Technology":-5.0,"Financials":-8.0,"Healthcare":-3.0,"Energy":-15.0,"Materials":-12.0}',
 '{"Technology":5.0,"Financials":2.0,"Healthcare":4.0,"Energy":-20.0,"Materials":-10.0}',
 53.29, 24.5, 'WEAK', 'China circuit breakers, EM contagion, commodity crash'),

-- 2016 recovery → trend_confirmation
('analog_2016_trump_reflation',  '2016-11-09', 'trend_confirmation',
 '{"rate_env":"rising","credit":"tightening","growth":"reflation","fiscal":"expected_stimulus"}',
 '{"Technology":3.0,"Financials":20.0,"Healthcare":-5.0,"Energy":10.0,"Industrials":12.0}',
 '{"Technology":8.0,"Financials":32.0,"Healthcare":-8.0,"Energy":20.0,"Industrials":18.0}',
 18.74, 26.0, 'STRONG', 'Trump election: reflation trade, financials and industrials led'),

-- 2018 Q4 selloff → transition_defensive
('analog_2018_q4_selloff',       '2018-12-24', 'transition_defensive',
 '{"rate_env":"peak_hiking","credit":"tightening","growth":"peak_cycle"}',
 '{"Technology":12.0,"Financials":8.0,"Healthcare":6.0,"Energy":5.0,"Utilities":4.0}',
 '{"Technology":22.0,"Financials":16.0,"Healthcare":12.0,"Energy":8.0,"Utilities":6.0}',
 36.07, 28.0, 'MODERATE', 'Fed overtightening fear: sharp Q4 correction, V-shaped recovery'),

-- 2019 QE pivot → trend_confirmation
('analog_2019_qe_pivot',         '2019-01-04', 'trend_confirmation',
 '{"rate_env":"pausing","credit":"easing","growth":"slowing_but_ok","fed":"dovish_pivot"}',
 '{"Technology":10.0,"Financials":8.0,"Healthcare":5.0,"Energy":7.0,"Industrials":9.0}',
 '{"Technology":28.0,"Financials":22.0,"Healthcare":15.0,"Energy":14.0,"Industrials":20.0}',
 21.43, 28.5, 'STRONG', 'Powell pivot: Fed signals rate pause, risk-on resumes'),

-- 2020 COVID crash → transition_defensive
('analog_2020_covid',            '2020-03-23', 'transition_defensive',
 '{"rate_env":"zero","credit":"extreme_stress","growth":"shutdown","event":"pandemic_shock"}',
 '{"Technology":20.0,"Financials":5.0,"Healthcare":15.0,"Energy":-20.0,"Utilities":5.0}',
 '{"Technology":55.0,"Financials":25.0,"Healthcare":30.0,"Energy":-10.0,"Utilities":10.0}',
 82.69, 30.0, 'DEGRADED', 'COVID trough: fastest bear market in history, tech led recovery'),

-- 2020 reflation → trend_confirmation
('analog_2020_reflation',        '2020-11-09', 'trend_confirmation',
 '{"rate_env":"zero","credit":"tight_spread","growth":"vaccine_recovery","fiscal":"massive_stimulus"}',
 '{"Technology":-2.0,"Financials":20.0,"Healthcare":5.0,"Energy":35.0,"Industrials":18.0}',
 '{"Technology":3.0,"Financials":35.0,"Healthcare":8.0,"Energy":55.0,"Industrials":28.0}',
 24.86, 33.0, 'STRONG', 'Vaccine announcement: great rotation to value and cyclicals'),

-- 2021 SPAC/meme peak → liquidity_fracture
('analog_2021_meme_peak',        '2021-02-12', 'liquidity_fracture',
 '{"rate_env":"rising_long_end","credit":"tight","growth":"stimulus_boom","retail":"speculative"}',
 '{"Technology":-8.0,"Financials":3.0,"Healthcare":-2.0,"Energy":15.0,"Utilities":-2.0}',
 '{"Technology":-5.0,"Financials":8.0,"Healthcare":2.0,"Energy":20.0,"Utilities":3.0}',
 19.97, 35.0, 'MODERATE', 'Meme stock peak: speculative excess in retail-driven names'),

-- 2022 Fed rate shock → transition_defensive
('analog_2022_rate_shock',       '2022-01-05', 'transition_defensive',
 '{"rate_env":"aggressive_hiking","credit":"widening","growth":"decelerating","inflation":"peak"}',
 '{"Technology":-18.0,"Financials":-6.0,"Healthcare":-4.0,"Energy":12.0,"Utilities":2.0}',
 '{"Technology":-28.0,"Financials":-15.0,"Healthcare":-8.0,"Energy":30.0,"Utilities":5.0}',
 23.22, 38.0, 'WEAK', 'Fed pivot to aggressive tightening: duration assets crushed'),

-- 2022 inflation peak → recovery_selective
('analog_2022_inflation_peak',   '2022-06-16', 'recovery_selective',
 '{"rate_env":"mid_hike_cycle","credit":"widening","growth":"decelerating","inflation":"peak"}',
 '{"Technology":12.0,"Financials":5.0,"Healthcare":6.0,"Energy":-5.0,"Utilities":4.0}',
 '{"Technology":15.0,"Financials":8.0,"Healthcare":10.0,"Energy":-10.0,"Utilities":6.0}',
 34.02, 28.0, 'MODERATE', 'Inflation peak: rate hike expectations plateau, bear market bounce'),

-- 2023 banking stress → transition_defensive
('analog_2023_svb',              '2023-03-10', 'transition_defensive',
 '{"rate_env":"high","credit":"bank_stress","growth":"ok","event":"bank_failures"}',
 '{"Technology":8.0,"Financials":-12.0,"Healthcare":3.0,"Energy":2.0,"Utilities":2.0}',
 '{"Technology":25.0,"Financials":-5.0,"Healthcare":5.0,"Energy":5.0,"Utilities":4.0}',
 26.52, 28.5, 'MODERATE', 'SVB/Signature failure: regional bank stress, flight to mega-cap tech'),

-- 2024 AI boom → trend_confirmation
('analog_2024_ai_boom',          '2024-01-08', 'trend_confirmation',
 '{"rate_env":"peak_hold","credit":"tight","growth":"resilient","theme":"ai_capex_cycle"}',
 '{"Technology":15.0,"Financials":5.0,"Healthcare":3.0,"Energy":2.0,"Industrials":6.0}',
 '{"Technology":28.0,"Financials":12.0,"Healthcare":6.0,"Energy":4.0,"Industrials":10.0}',
 12.45, 34.0, 'STRONG', 'Nvidia/AI cycle: semiconductor and cloud led broad market')

ON CONFLICT (analog_id) DO NOTHING;
