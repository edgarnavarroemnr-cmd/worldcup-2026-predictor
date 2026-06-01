-- ============================================================================
-- World Cup 2026 Predictor — PostgreSQL Schema & Analytical Queries
-- ============================================================================
-- Author: Edgar Navarro | Bilingual Data Analyst
-- Data source: Fjelstul World Cup Database (CC-BY-SA 4.0), men's 1930-2022
--
-- This script (1) defines a clean relational schema, (2) loads the cleaned
-- match data, and (3) answers real analytical questions with window
-- functions, CTEs, and aggregations — the SQL backbone of the project.
--
-- USAGE (local PostgreSQL, e.g. via psql):
--   createdb worldcup
--   psql worldcup -f worldcup_postgres.sql
--   \copy matches FROM 'output/matches.csv' WITH CSV HEADER
-- ============================================================================

DROP TABLE IF EXISTS matches CASCADE;

CREATE TABLE matches (
    match_id          INTEGER PRIMARY KEY,
    tournament_name   TEXT    NOT NULL,
    year              INTEGER NOT NULL,
    stage_name        TEXT,
    match_date        DATE,
    home_team_name    TEXT    NOT NULL,
    away_team_name    TEXT    NOT NULL,
    home_team_score   INTEGER,
    away_team_score   INTEGER,
    result            TEXT
);

CREATE INDEX idx_matches_year ON matches(year);
CREATE INDEX idx_matches_home ON matches(home_team_name);
CREATE INDEX idx_matches_away ON matches(away_team_name);

-- Load with: \copy matches FROM 'output/matches.csv' WITH CSV HEADER

-- ----------------------------------------------------------------------------
-- VIEW: team_match_long — one row per team per match (the analytical base)
-- ----------------------------------------------------------------------------
CREATE OR REPLACE VIEW team_match_long AS
SELECT match_id, year, tournament_name, stage_name, match_date,
       home_team_name AS team, away_team_name AS opponent,
       home_team_score AS goals_for, away_team_score AS goals_against,
       CASE WHEN home_team_score > away_team_score THEN 1 ELSE 0 END AS win,
       CASE WHEN home_team_score = away_team_score THEN 1 ELSE 0 END AS draw
FROM matches
UNION ALL
SELECT match_id, year, tournament_name, stage_name, match_date,
       away_team_name, home_team_name,
       away_team_score, home_team_score,
       CASE WHEN away_team_score > home_team_score THEN 1 ELSE 0 END,
       CASE WHEN away_team_score = home_team_score THEN 1 ELSE 0 END
FROM matches;

-- ============================================================================
-- ANALYTICAL QUERIES
-- ============================================================================

-- Q1. All-time leaderboard: wins, win rate, goal differential per nation
--     (min. 10 matches to filter out one-off appearances)
SELECT team,
       COUNT(*)                                   AS matches,
       SUM(win)                                   AS wins,
       ROUND(AVG(win)::numeric, 3)                AS win_rate,
       ROUND(AVG(goals_for)::numeric, 2)          AS gf_per_match,
       ROUND(AVG(goals_against)::numeric, 2)      AS ga_per_match,
       ROUND((AVG(goals_for) - AVG(goals_against))::numeric, 2) AS goal_diff_avg
FROM team_match_long
GROUP BY team
HAVING COUNT(*) >= 10
ORDER BY win_rate DESC, goal_diff_avg DESC
LIMIT 20;

-- Q2. Mexico's tournament-by-tournament performance with running totals
--     (window function: cumulative wins across Mexico's WC history)
WITH mex AS (
    SELECT year,
           SUM(win)               AS wins,
           COUNT(*)               AS matches,
           SUM(goals_for)         AS gf,
           SUM(goals_against)     AS ga
    FROM team_match_long
    WHERE team = 'Mexico'
    GROUP BY year
)
SELECT year, matches, wins, gf, ga,
       SUM(wins)    OVER (ORDER BY year) AS cumulative_wins,
       SUM(matches) OVER (ORDER BY year) AS cumulative_matches,
       ROUND(AVG(wins) OVER (ORDER BY year
              ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)::numeric, 2)
                                          AS wins_3tourney_moving_avg
FROM mex
ORDER BY year;

-- Q3. Head-to-head record between any two nations (parameterized example)
--     Brazil vs Argentina — the classic rivalry
SELECT team, opponent,
       COUNT(*)                        AS meetings,
       SUM(win)                        AS wins,
       SUM(draw)                       AS draws,
       COUNT(*) - SUM(win) - SUM(draw) AS losses
FROM team_match_long
WHERE team = 'Brazil' AND opponent = 'Argentina'
GROUP BY team, opponent;

-- Q4. Highest-scoring tournaments: goals per match trend over time
SELECT year, tournament_name,
       COUNT(DISTINCT match_id)                          AS matches,
       SUM(goals_for) / 2                                AS total_goals,
       ROUND((SUM(goals_for)::numeric / COUNT(DISTINCT match_id)) / 2 * 2, 2)
                                                         AS goals_per_match
FROM team_match_long
GROUP BY year, tournament_name
ORDER BY year;

-- Q5. Biggest "giant killers": teams ranked by wins in knockout stages
WITH knockouts AS (
    SELECT * FROM team_match_long
    WHERE stage_name NOT ILIKE '%group%'
)
SELECT team,
       COUNT(*)                        AS knockout_matches,
       SUM(win)                        AS knockout_wins,
       ROUND(AVG(win)::numeric, 3)     AS knockout_win_rate,
       RANK() OVER (ORDER BY SUM(win) DESC) AS rank_by_ko_wins
FROM knockouts
GROUP BY team
HAVING COUNT(*) >= 5
ORDER BY knockout_wins DESC
LIMIT 15;
