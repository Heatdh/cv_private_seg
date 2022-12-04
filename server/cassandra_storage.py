from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider


def create_connection():
    auth_provider = PlainTextAuthProvider(username='cassandra', password='cassandra')
    cluster = Cluster(['db'])#, auth_provider=auth_provider)

    session = cluster.connect()
    session.execute("""
            CREATE KEYSPACE IF NOT EXISTS test_keyspace
            WITH REPLICATION =
            { 'class' : 'SimpleStrategy', 'replication_factor' : 1 }
            """)
    session.set_keyspace('test_keyspace')
    return session


def create_database(cursor):
    cursor.execute("CREATE TABLE IF NOT EXISTS data(image_name text, content blob, PRIMARY KEY (image_name));")


def insert_data(cursor, image_64_encode, image_name):
    strCQL = "INSERT INTO data (image_name,content) VALUES (?,?);"
    pStatement = cursor.prepare(strCQL)
    cursor.execute(pStatement, [image_name, image_64_encode])


def get_specific_data(cursor, image_name):
    res = cursor.execute("SELECT image_name,content FROM data WHERE image_name=\'{}\';".format(image_name))
    print(res)
    return res[0].content


def get_images(cursor):
    res = (cursor.execute("SELECT image_name FROM data;"))
    return [el.image_name for el in res]


def delete_specific_image(cursor, image_name):
    (cursor.execute("DELETE FROM data WHERE image_name=\'{}\'").format(image_name))
