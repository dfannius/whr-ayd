import sqlite3

def make_db():
    con = sqlite3.connect("whr.db")
    cur = con.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS games(date INTEGER, winner TEXT, loser TEXT, UNIQUE(date, winner, loser))")
    cur.execute("CREATE TABLE IF NOT EXISTS ratings(player TEXT, date INTEGER, rating REAL, std REAL, UNIQUE(player, date))")
    cur.execute("DROP TABLE IF EXISTS yd_ratings")
    cur.execute("CREATE TABLE IF NOT EXISTS yd_ratings(player TEXT, date INTEGER, yd_rating INTEGER, UNIQUE(player, date))")
    con.commit()

make_db()
