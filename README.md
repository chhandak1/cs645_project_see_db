# CS 645 Project: SeeDB

### Instructions to load the data and create tables

1. Download the data and unzip it.

2. Create a new database in PostgreSQL called ``census``.

3. Load data and create tables:

* Change the folder path in ``queries/create_tables.sql`` to the full path for the ``adult.data`` file.


* After that run: ``psql -f create_tables.sql -U postgres census``. This creates the table ``AdultData`` and also two other tables ``married`` and ``unmarried``.


### Instructions to run the code:

1. Install psycopg2:
```
pip install psycopg2

pip install psycopg2-binary 
```

2. Add your username and password to make the connection (lines 24 and 25).

3. Run the code using: ``python run_aggregation.py``
