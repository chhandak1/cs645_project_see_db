SET client_encoding TO 'UTF8';
drop table IF EXISTS AdultData cascade;

create table AdultData (age int, workclass text, fnlwgt int, education text, education_num text, marital_status text, occupation text, relationship text, race text, sex text, capital_gain int, capital_loss int, hours_per_week int, native_country text, income text);

\COPY AdultData (age, workclass, fnlwgt, education, education_num, marital_status, occupation, relationship, race, sex, capital_gain, capital_loss, hours_per_week, native_country, income) FROM '/Users/hadi/Hadi Drive/UMass Amherst/Semester 8/645-DB Design & Imp./Project/census+income/adult.data.csv' WITH DELIMITER ',' NULL AS '';

alter table AdultData add row_id serial;

select * from AdultData limit 10;

-- drop table IF EXISTS unmarried cascade;

create table unmarried as
select * 
from AdultData
where marital_status= ' Separated' 
or marital_status= ' Never-married'
or marital_status= ' Widowed'
or marital_status= ' Divorced';

-- drop table IF EXISTS married cascade;

create table married as
select * 
from AdultData
where marital_status= ' Married-civ-spouse' 
or marital_status= ' Married-AF-spouse'
or marital_status= ' Married-spouse-absent';