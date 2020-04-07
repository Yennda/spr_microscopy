import sqlite3

class Table():
    def __init__(self, connection, name):
        self.con = connection
        self._name = name
        
        columns = self.con.execute("""
        PRAGMA table_info('{}')
        """.format(name))
        self._columns = [c[1] for c in columns]
        
    @property
    def columns(self):
        return str(self._columns[1:])[1:-1].replace("'", "")
    
    @property
    def name(self):
        return "'{}'".format(self._name)
    
    @staticmethod
    def process_values(*args):
        values = str()
        for a in args:
            if type(a) is int or type(a) is float:
                values += str(a)
            elif type(a) is str:
                values += "'{}'".format(a)
            values += ', '
        return values[:-2]
        
    def insert(self, *args):
        values = self.process_values(*args)      
        self.con.execute("""
        INSERT INTO  {} ({})
        VALUES ({})
        """.format(self.name, self.columns, values))
    
    def show_all(self):
        data = self.con.execute("""
        SELECT * FROM {};
        """.format(self.name))

        print('ID\t' + self.columns.replace(', ', '\t'))

        for row in data:
            print(str(row).replace(', ', '\t').replace("'", "")[1:-1])
            
    def commit(self):
        self.con.commit()
        
    def clear_all(self):
        delete = input('Really? y/n')
        
        if delete == 'y':
            self.con.execute("""
            DELETE FROM {};
            """.format(self.name))
        