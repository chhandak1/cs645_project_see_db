import math
import psycopg2
import numpy as np
import pandas as pd

### INITIALIZERS ###

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

k = 1
num_partitions = 10 ##Number of iterations/splits
confidence_level = 0.95

cumulative_sums = {}
count_entries = {}
running_means = {}

#### METHODS ####

def execute_query_and_get_rows(connection, query):
    cur = connection.cursor()
    cur.execute(query)
    col_names = [desc[0] for desc in cur.description]
    rows = cur.fetchall()
    cur.close()

    return rows, col_names

def fetch_grouped_data(table_name, lower_bound, upper_bound, a, continous_vars):

    # assert (a in categorical_variables)

    inside_query = f'(select '
    query = f'select t.g1, '

    for grouping_attribute in a:
        inside_query += f'{grouping_attribute}, '
        query += f't.{grouping_attribute}, '

    for continuous_var in continous_vars:
         inside_query += f'{continuous_var}, '

    inside_query = inside_query[:-2] + ", case when marital_status!=' Never-married' then 'target' else 'reference' end as g1"
    inside_query += f" from {table_name} where row_id between {lower_bound} and {upper_bound})  as t"

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
    
    #print(query)
    
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
            # print(i,j)
            selected_data_dict[col_names[i+aggregate_start_idx]][attribute_types[j]] = selected_data[j, i]
    return selected_data_dict

def get_probability_distribution(attribute):
    probability_distribution = np.array([x / sum(attribute) for x in attribute], dtype=np.float32)

    return probability_distribution

def kl_divergence(target, reference):
    # epsilon is smallest positive float
    epsilon = np.finfo(float).eps

    # ensuring both target and reference values are between 'epsilon' and '1.0' to prevent divide by 0 errors.
    # target = np.clip(target, epsilon, 1.0)
    # reference = np.clip(reference, epsilon, 1.0)
    
    kl_divergence_value = np.sum(target * np.log(target/reference))

    return kl_divergence_value

def sort_kl_divergences(kl_divergences):
    sorted_kl_divergences = sorted(kl_divergences.items(), key = lambda item: item[1], reverse=True)

    return dict(sorted_kl_divergences)

def get_top_k_divergences(kl_diveregences, k):
    top_k_items = sorted(kl_diveregences.items(), key = lambda item: item[1], reverse=True)[:k]
    
    return dict(top_k_items)

def get_database_size(connection):
    cur = connection.cursor()
    query = 'select max(row_id) from AdultData;'
    cur.execute(query)
    col_names = [desc[0] for desc in cur.description]
    rows = cur.fetchall()
    # print(rows)
    table_size = rows[0][0]
    cur.close()

    return table_size

def get_running_means(kld_values):
    for key, value in kld_values.items():
        # if key in cumulative_sums:
            # cumulative_sums[key] += value
            # count_entries[key] += 1
        running_means[key] = np.mean(value)
        # else:
        #     running_means[key] = np.nan
            # cumulative_sums[key] = value
            # count_entries[key] = 1
        
        # running_means[key] = cumulative_sums[key] / count_entries[key]

    return running_means

def hoeffding_serfling_confidence_interval(Y, N, delta):
    """
    These are named based on the paper.
    From paper: "each value in Y corresponds to an estimate of utility computed based on the records seen so far."
    So, we probbaly need to call this function for each view in each phase?!
    """
    m = len(Y)
    mean = sum(Y) / m
    
    # Let's find the confidence interval
    temp = (1 - (m - 1) / N)
    epsilon = math.sqrt((temp * (2 * math.log(math.log(m) + math.log(math.pi**2 / (3 * delta))))) / (2 * m))
    
    # Let's find lower and upper bounds
    lower_bound = mean - epsilon
    upper_bound = mean + epsilon

    return mean, lower_bound, upper_bound

def hoeffding_serfling_inequality(m, n, delta):
    epsilon = math.sqrt((1 - (m - 1) / n) * (math.log(2 / delta) / (2 * m)))
    return epsilon

### RUN PROGRAM ###

list_of_list_of_groupby_attributes = [categorical_variables]


table_size = get_database_size(conn)

partition_size = np.ceil(table_size/num_partitions)

kld_values = {}
pruning_list = set()

for i in range(num_partitions):
    
    mean_kld = {}
    lower_bound_kld = {}
    upper_bound_kld = {}

    #print(f'RUN {i+1}:')
    lower_bound = round(i * partition_size)
    upper_bound = round(lower_bound + partition_size)
    for list_of_groupby_attributes in list_of_list_of_groupby_attributes:
        data, col_names = fetch_grouped_data(table_name='AdultData', lower_bound=lower_bound, upper_bound=upper_bound, a=list_of_groupby_attributes, continous_vars=continuous_variables)
        reference_array, target_array = split_into_reference_and_target(data)
        # print(reference_array.shape)
        # print(target_array.shape)

    # df = pd.DataFrame(data, columns=col_names)
    # df.to_csv('check2.csv')    

    # print(data)

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
            #print(key1)
            #print(missing_keys_in_target, missing_keys_in_reference)
            for missing_key_in_target in missing_keys_in_target:
                dict_of_prob_distribution_target[key1][key2][missing_key_in_target] = 0
            for missing_key_in_reference in missing_keys_in_reference:
                dict_of_prob_distribution_reference[key1][key2][missing_key_in_reference] = 0

    # all_ref_prob_dist = {}
    # all_target_prob_dist = {}
    # all_kl_divergences = {}
   
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

        for key in target_prob_dist:
            kl_value = kl_divergence(target_prob_dist[key], ref_prob_dist[key])
            if np.isnan(kl_value) or np.isinf(kl_value):
                kl_value = 0 
            # if np.isinf(kl_value):
            #     kl_value = np.nan
            if f'{key1}_{key}' in kld_values.keys():
                kld_values[f'{key1}_{key}'].append(kl_value)
            else:
                kld_values[f'{key1}_{key}'] = [kl_value]

        print(kld_values)
        # print(kld_values)
        # running_kld_mean = get_running_means(kld_values)
        # print(running_kld_mean)
    # sorted_kld_values = sort_kl_divergences(running_kld_mean)
    # max_kld_value = list(sorted_kld_values.values())[0]
    
    delta = 1 - confidence_level
    for key_kld, list_kld_values in kld_values.items():
        mean_kld[key_kld], lower_bound_kld[key_kld], upper_bound_kld[key_kld] = hoeffding_serfling_confidence_interval(list_kld_values, partition_size, delta)
    
    sorted_kld_values = sort_kl_divergences(mean_kld)
    top_k_kld_keys = list(sorted_kld_values.keys())[:k]
    
    kth_kld_key = list(sorted_kld_values.keys())[k-1]
    lower_bound_kth_kld_value = lower_bound_kld[kth_kld_key]
    print("Lower bound:", lower_bound_kth_kld_value)
    for key_kld, upper_bound_kld_val in upper_bound_kld.items():
        if key_kld not in top_k_kld_keys:
            if upper_bound_kld_val<lower_bound_kth_kld_value:
                pruning_list.add(key_kld)
    print(len(list(sorted_kld_values.keys())))
    print("Top k:",top_k_kld_keys)
    print("Pruned:",pruning_list)
    print()
    # # print(epsilon_m, lower_bound, upper_bound)

    # # top_k_diveregences = get_top_k_divergences(running_kld_mean, k)
    # # print(f'Top {k} highest utilities:')
    # # for i in top_k_diveregences:
    # #    print(f"{i} = {top_k_diveregences[i]:.4f}")

conn.close()