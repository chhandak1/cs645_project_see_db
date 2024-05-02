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

continuous_variables = ['age',
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
    for continuous_variable in continuous_variables:
        if continuous_variable!=a:
            query += f'''sum({continuous_variable}) as sum_{continuous_variable}, 
            avg({continuous_variable}) as avg_{continuous_variable}, '''
    query = query[:-2]

    query += f' from {table_name} group by {a} order by {a}'

    rows_fetched, col_names = execute_query_and_get_rows(conn, query)

    return np.asarray(rows_fetched), col_names

def get_probability_distribution(attribute):
    probability_distribution = np.array([x / sum(attribute) for x in attribute], dtype=np.float32)

    return probability_distribution

def kl_divergence(target, reference):
    kl_divergence_value = np.sum(target * np.log(target/reference))

    return kl_divergence_value

def get_top_k_divergences(kl_diveregences, k):
    top_k_items = sorted(kl_diveregences.items(), key = lambda item: item[1], reverse=True)[:k]
    
    return dict(top_k_items)

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

## 1. dict_of_prob_distribution_xxxx contains key value pairs. Need to normalize the values in [0,1] range such that they add up to 1.
ref_prob_dist = {}
for m_ref in list(dict_of_prob_distribution_married.keys())[1:]: # excluded the count from the probability distributions
    ref_prob_dist[f'{m_ref}'] = get_probability_distribution(dict_of_prob_distribution_married[m_ref])

target_prob_dist = {}
for m_target in list(dict_of_prob_distribution_unmarried.keys())[1:]: # excluded the count from the probability distributions
    target_prob_dist[f'{m_target}'] = get_probability_distribution(dict_of_prob_distribution_unmarried[m_target])

## 2. Calculate KL-divergence for each key between dict_of_prob_distribution_married and dict_of_prob_distribution_unmarried.
kld_values = {}
for key in target_prob_dist:
    kld_values[key] = kl_divergence(target_prob_dist[key], ref_prob_dist[key])
    #print(f"KL-Divergence of '{key}' between Unmarried and Married people is = {kld_values[key]:.4f}")

k = 5
top_k_diveregences = get_top_k_divergences(kld_values, k)
print(f'Top {k} highest utilities:')
for i in top_k_diveregences:
    print(f"{i} = {top_k_diveregences[i]:.4f}")

## TODOs
## 3. Confidence interval calculation using Hoeffding-Serfling inequality.