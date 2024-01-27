import sqlite3
import sys

def run(name):
    con = sqlite3.connect("whr.db")
    cur = con.cursor()
    cur.execute(f"DELETE FROM games WHERE winner='{name}'")
    cur.execute(f"DELETE FROM games WHERE loser='{name}'")
    cur.execute(f"DELETE FROM ratings WHERE player='{name}'")
    cur.execute(f"DELETE FROM yd_ratings WHERE player='{name}'")
    con.commit()

run(sys.argv[1])    
