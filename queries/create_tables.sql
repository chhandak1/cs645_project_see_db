SET client_encoding TO 'UTF8';
drop table IF EXISTS AdultData cascade;

create table AdultData (age int, workclass text, fnlwgt int, education text, education_num text, marital_status text, occupation text, relationship text, race text, sex text, capital_gain	int, capital_loss int, hours_per_week int, native_country text, income text);

\COPY AdultData (age, workclass, fnlwgt, education, education_num, marital_status, occupation, relationship, race, sex, capital_gain, capital_loss, hours_per_week, native_country, income) FROM 'C:\\Users\\chhan\\Documents\\UMass Courses\\CS 645 Databases\\Project\\data\\adult.data.csv' WITH DELIMITER ',' NULL AS '';

create table unmarried as
select * 
from AdultData
where marital_status= ' Separated' 
or marital_status= ' Never-married'
or marital_status= ' Widowed'
or marital_status= ' Divorced';

create table married as
select * 
from AdultData
where marital_status= ' Married-civ-spouse' 
or marital_status= ' Married-AF-spouse'
or marital_status= ' Married-spouse-absent';




