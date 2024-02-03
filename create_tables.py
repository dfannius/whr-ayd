import sqlite3

def make_db():
    con = sqlite3.connect("whr.db")
    cur = con.cursor()

    cur.execute("DROP TABLE IF EXISTS games")
    cur.execute("CREATE TABLE IF NOT EXISTS games(date INTEGER, winner_id INTEGER, loser_id INTEGER, UNIQUE(date, winner_id, loser_id))")

    cur.execute("DROP TABLE IF EXISTS ratings")
    cur.execute("CREATE TABLE IF NOT EXISTS ratings(player_id INTEGER, date INTEGER, rating REAL, std REAL, UNIQUE(player_id, date))")

    cur.execute("DROP TABLE IF EXISTS yd_ratings")
    cur.execute("CREATE TABLE IF NOT EXISTS yd_ratings(player_id INTEGER, date INTEGER, yd_rating INTEGER, UNIQUE(player_id, date))")

    cur.execute("DROP TABLE IF EXISTS ids")
    cur.execute("CREATE TABLE IF NOT EXISTS ids(player_id INTEGER, handle TEXT, UNIQUE(player_id))")

    con.commit()

make_db()
