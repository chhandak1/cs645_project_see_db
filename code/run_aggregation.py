import psycopg2
import numpy as np
import pandas as pd

categorical_variables = ['workclass', 
                       'education', 
                    #    'marital_status', 
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
    user="postgres", ## Add your SQL username here (postgres by default)
    password="C!sco123" ## Add your SQL password here
)

def execute_query_and_get_rows(connection, query):
    cur = connection.cursor()
    cur.execute(query)
    col_names = [desc[0] for desc in cur.description]
    rows = cur.fetchall()
    cur.close()

    return rows, col_names

def fetch_grouped_data(table_name, a, continous_vars):

    # assert (a in categorical_variables)

    inside_query = f'(select '
    query = f'select t.g1, '

    for grouping_attribute in a:
        inside_query += f'{grouping_attribute}, '
        query += f't.{grouping_attribute}, '

    for continuous_var in continous_vars:
         inside_query += f'{continuous_var}, '

    inside_query = inside_query[:-2] + ", case when marital_status!=' Never-married' then 'target' else 'reference' end as g1"
    inside_query += f" from {table_name})  as t"

    query += f'count(*) as count, '
    for continuous_variable in continuous_variables:
        if continuous_variable!=a:
            # query += f'''sum(t.{continuous_variable}) as sum_{continuous_variable}, 
            # avg(t.{continuous_variable}) as avg_{continuous_variable}, '''
            query += f'''avg(t.{continuous_variable}) as avg_{continuous_variable}, '''
            
    query = query[:-2]

    query += f" from {inside_query} group by grouping sets("
    for grouping_attribute in a:
        query += f'(t.{grouping_attribute}, t.g1), '
    query = query[:-2] + ') order by t.g1'
    
    print(query)
    
    rows_fetched, col_names = execute_query_and_get_rows(conn, query)

    return np.asarray(rows_fetched), col_names

def split_into_reference_and_target(data):
    mask = data[:, 0]=='reference'
    reference_data = data[mask]
    target_data = data[~mask]

    return reference_data, target_data

def get_data_for_col_name(data, col_names, col_num, aggregate_start_idx):
    
    selected_data_dict = {}
    mask = data[:, col_num]!=None
    mask_offset = list(mask).index(True)
    attribute_types = data[mask, col_num]
    selected_data = data[mask, aggregate_start_idx:]

    for i in range(selected_data.shape[1]):
        selected_data_dict[col_names[i+aggregate_start_idx]] = {}
        for j in range(len(attribute_types)):
            print(i,j)
            selected_data_dict[col_names[i+aggregate_start_idx]][attribute_types[j]] = selected_data[j, i]
    return selected_data_dict

def get_probability_distribution(attribute):
    probability_distribution = np.array([x / sum(attribute) for x in attribute], dtype=np.float32)

    return probability_distribution

def kl_divergence(target, reference):
    kl_divergence_value = np.sum(target * np.log(target/reference))

    return kl_divergence_value

def get_top_k_divergences(kl_diveregences, k):
    top_k_items = sorted(kl_diveregences.items(), key = lambda item: item[1], reverse=True)[:k]
    
    return dict(top_k_items)


list_of_list_of_groupby_attributes = [categorical_variables]

for list_of_groupby_attributes in list_of_list_of_groupby_attributes:
    data, col_names = fetch_grouped_data(table_name='AdultData', a=list_of_groupby_attributes, continous_vars=continuous_variables)
    reference_array, target_array = split_into_reference_and_target(data)
    print(reference_array.shape)
    print(target_array.shape)

df = pd.DataFrame(data, columns=col_names)
df.to_csv('check2.csv')

conn.close()

print(data)

dict_of_prob_distribution_target = {}
dict_of_prob_distribution_reference = {}

aggregate_start_idx = col_names.index('count')

for i in range(1, aggregate_start_idx):
    dict_of_prob_distribution_target[col_names[i]] = get_data_for_col_name(target_array, col_names,  col_num=i, aggregate_start_idx=aggregate_start_idx)
    dict_of_prob_distribution_reference[col_names[i]] = get_data_for_col_name(reference_array, col_names, col_num=i, aggregate_start_idx=aggregate_start_idx)

# print(dict_of_prob_distribution_target)


for key1, value1 in dict_of_prob_distribution_target.items():
    for key2, value2 in dict_of_prob_distribution_target[key1].items():
        target_keys = set(list(dict_of_prob_distribution_target[key1][key2].keys()))
        reference_keys = set(list(dict_of_prob_distribution_reference[key1][key2].keys()))
        missing_keys_in_target = reference_keys.difference(target_keys)
        missing_keys_in_reference = target_keys.difference(reference_keys)
        print(key1)
        print(missing_keys_in_target, missing_keys_in_reference)
        for missing_key_in_target in missing_keys_in_target:
            dict_of_prob_distribution_target[key1][key2][missing_key_in_target] = 0
        for missing_key_in_reference in missing_keys_in_reference:
            dict_of_prob_distribution_reference[key1][key2][missing_key_in_reference] = 0

all_ref_prob_dist = {}
all_target_prob_dist = {}
all_kl_divergences = {}
for key1, value1 in dict_of_prob_distribution_target.items():
    ref_prob_dist = {}

    for m_ref in list(dict_of_prob_distribution_target[key1].keys())[1:]: # excluded the count from the probability distributions
        list_of_values_ref = []
        list_of_labels = []
        for label, aggregate in dict_of_prob_distribution_target[key1][m_ref].items():
            list_of_labels.append(label)
            list_of_values_ref.append(aggregate)
        ref_prob_dist[f'{m_ref}'] = get_probability_distribution(list_of_values_ref)

    target_prob_dist = {}
    for m_target in list(dict_of_prob_distribution_reference[key1].keys())[1:]: # excluded the count from the probability distributions
        list_of_values_target = []
        for label in list_of_labels:
            list_of_values_target.append(dict_of_prob_distribution_reference[key1][m_target][label])
        target_prob_dist[f'{m_target}'] = get_probability_distribution(list_of_values_target)

    ## 2. Calculate KL-divergence for each key between dict_of_prob_distribution_married and dict_of_prob_distribution_unmarried.
    kld_values = {}
    for key in target_prob_dist:
        kld_values[f'{key1}_{key}'] = kl_divergence(target_prob_dist[key], ref_prob_dist[key])
    #print(f"KL-Divergence of '{key}' between Unmarried and Married people is = {kld_values[key]:.4f}")

k = 5
top_k_diveregences = get_top_k_divergences(kld_values, k)
print(f'Top {k} highest utilities:')
for i in top_k_diveregences:
    print(f"{i} = {top_k_diveregences[i]:.4f}")

# ## TODOs
# ## 3. Confidence interval calculation using Hoeffding-Serfling inequality.