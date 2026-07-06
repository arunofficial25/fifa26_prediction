CREATE DATABASE WC2026Predictor;
USE WC2026Predictor;

CREATE TABLE Teams (
    TeamID INT AUTO_INCREMENT PRIMARY KEY,
    TeamName VARCHAR(100) NOT NULL UNIQUE,
    Confederation VARCHAR(50),
    IsHost BOOLEAN DEFAULT 0
);

CREATE TABLE Matches (
    MatchID INT AUTO_INCREMENT PRIMARY KEY,
    MatchDate DATE NOT NULL,
    HomeTeamID INT NOT NULL,
    AwayTeamID INT NOT NULL,
    HomeGoals INT,
    AwayGoals INT,
    Tournament VARCHAR(100),
    Stage VARCHAR(50),
    IsNeutralVenue BOOLEAN DEFAULT 0,
    IsWC2026 BOOLEAN DEFAULT 0,
    FOREIGN KEY (HomeTeamID) REFERENCES Teams(TeamID),
    FOREIGN KEY (AwayTeamID) REFERENCES Teams(TeamID)
);

CREATE TABLE FIFARankings (
    RankingID INT AUTO_INCREMENT PRIMARY KEY,
    TeamID INT NOT NULL,
    RankDate DATE NOT NULL,
    `Rank` INT,
    Points DECIMAL(8,2),
    FOREIGN KEY (TeamID) REFERENCES Teams(TeamID)
);

CREATE TABLE EloRatings (
    EloID INT AUTO_INCREMENT PRIMARY KEY,
    TeamID INT NOT NULL,
    RatingDate DATE NOT NULL,
    EloRating DECIMAL(8,2),
    FOREIGN KEY (TeamID) REFERENCES Teams(TeamID)
);

CREATE TABLE Fixtures2026 (
    FixtureID INT AUTO_INCREMENT PRIMARY KEY,
    MatchDate DATE NOT NULL,
    HomeTeamID INT,
    AwayTeamID INT,
    Stage VARCHAR(50),
    Venue VARCHAR(100),
    IsCompleted BOOLEAN DEFAULT 0,
    ActualResult VARCHAR(10),
    FOREIGN KEY (HomeTeamID) REFERENCES Teams(TeamID),
    FOREIGN KEY (AwayTeamID) REFERENCES Teams(TeamID)
);

CREATE TABLE Predictions (
    PredictionID INT AUTO_INCREMENT PRIMARY KEY,
    FixtureID INT NOT NULL,
    ModelName VARCHAR(50),
    HomeWinProb DECIMAL(5,4),
    DrawProb DECIMAL(5,4),
    AwayWinProb DECIMAL(5,4),
    PredictedOutcome VARCHAR(10),
    PredictionDate DATETIME DEFAULT CURRENT_TIMESTAMP,
    WasCorrect BOOLEAN NULL,
    FOREIGN KEY (FixtureID) REFERENCES Fixtures2026(FixtureID)
);

SHOW TABLES;


USE WC2026Predictor;

ALTER TABLE Teams 
MODIFY TeamName VARCHAR(100) 
CHARACTER SET utf8mb4 
COLLATE utf8mb4_0900_as_cs 
NOT NULL;

SET FOREIGN_KEY_CHECKS = 0;
TRUNCATE TABLE Matches;
TRUNCATE TABLE Teams;
SET FOREIGN_KEY_CHECKS = 1;

TRUNCATE TABLE Matches;
TRUNCATE TABLE Teams;

SELECT COUNT(*) AS TeamCount FROM Teams;
SELECT COUNT(*) AS MatchCount FROM Matches;
SELECT COUNT(*) AS RankingCount FROM FIFARankings;

-- spot check a few real matches
SELECT m.MatchDate, t1.TeamName AS Home, t2.TeamName AS Away, m.HomeGoals, m.AwayGoals, m.Tournament
FROM Matches m
JOIN Teams t1 ON m.HomeTeamID = t1.TeamID
JOIN Teams t2 ON m.AwayTeamID = t2.TeamID
ORDER BY m.MatchDate DESC
LIMIT 10;