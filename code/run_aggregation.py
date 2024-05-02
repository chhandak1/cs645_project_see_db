import psycopg2
import numpy as np

categorical_variables = ['workclass', 
                       'education', 
                       'education_num', 
                       'marital_status', 
                       'occupation', 
                       'relationship',
                       'race',
                       'sex', 
                       'native_country', 
                       'income']

continuos_variables = ['age',
                       'fnlwgt',
                       'capital_gain',
                       'capital_loss',
                       'hours_per_week',]

conn = psycopg2.connect(
    host="localhost",
    database="census",
    user="", ## Add your SQL username here (postgres by default)
    password="" ## Add your SQL password here
)

def execute_query_and_get_rows(connection, query):
    cur = connection.cursor()
    cur.execute(query)
    col_names = [desc[0] for desc in cur.description]
    rows = cur.fetchall()
    cur.close()

    return rows, col_names

def fetch_grouped_data(table_name, a):

    assert (a in categorical_variables)

    query = f'select {a}, '
    # for categorical_variable in categorical_variables:
        # if categorical_variable!=a:
        #     query += f'count({categorical_variable}) as count_{categorical_variable}, '
    query += f'count(*) as count, '
    for continuos_variable in continuos_variables:
        if continuos_variable!=a:
            query += f'''sum({continuos_variable}) as sum_{continuos_variable}, 
            avg({continuos_variable}) as avg_{continuos_variable}, '''
    query = query[:-2]

    query += f' from {table_name} group by {a} order by {a}'

    rows_fetched, col_names = execute_query_and_get_rows(conn, query)

    return np.asarray(rows_fetched), col_names

married_data_grouped_by_a, col_names = fetch_grouped_data(table_name='married', a='sex')
unmarried_data_grouped_by_a, _ = fetch_grouped_data(table_name='unmarried', a='sex')

conn.close()

print("Column names:", col_names)
print("Married:", married_data_grouped_by_a)
print("Unmarried:", unmarried_data_grouped_by_a)

dict_of_prob_distribution_married = {}
dict_of_prob_distribution_unmarried = {}

for i in range(1, len(col_names)):
    dict_of_prob_distribution_married[col_names[i]] = married_data_grouped_by_a[:, i]
    dict_of_prob_distribution_unmarried[col_names[i]] = unmarried_data_grouped_by_a[:, i]

print("Married:", dict_of_prob_distribution_married)
print("Unmarried:", dict_of_prob_distribution_unmarried)

## TODOs
## 1. dict_of_prob_distribution_xxxx contains key value pairs. Need to normalize the values in [0,1] range such that they add up to 1.
## 2. Calculate KL-divergence for each key between dict_of_prob_distribution_married and dict_of_prob_distribution_unmarried.
## 3. Confidence interval calculation using Hoeffding-Serfling inequality.