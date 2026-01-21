import mysql.connector # type: ignore

def connect_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password=" ",
        database=" "
    )

def get_answer(query):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT answer FROM queries WHERE question = %s", (query,))
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None

