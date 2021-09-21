# Bank-Data-Anomaly-Detection

-------------

## Dataset

The original data consist of  116208 records  each including 8 attributes: `Account No`, `DATE`, `TRANSACTION DETAILS`, `CHQ.NO.`, `VALUE DATE`, `WITHDRAWAL AMT`, `DEPOSIT AMT`, `BALANCE AMT` with lots of `Null` values. 

## Preprocessing Stage

* Normalizing numeric attributes
* Filling “empty” attributes based on some convention (e.g. most common element)
* Erasing “unreasonable” spaces at the beginning and the end of attribute values;
* Making all the strings lower-case
* Deleting columns of high variability (let us say that column is of high variability if 70% of the
  rows have unique value).
* Deleting all columns that contains some date format and reformulate the column
  accordingly.
* Fixing date format for all the rows
* Splitting the columns containing both text and number 
* Dropping duplicated transactions, if 70% of their attributes are equal

## Categorical Anomaly Detection 

The Isolation Forest has been used for detection the anomaly in dataset. 

## Contents

- [`BankDataAnomalyDetection`](BankDataAnomalyDetection.py) - All the pre-processing and analysis steps have been accumulated in.
- `bank.xlsx` - Bank Data

## TODO

* Using other approaches to detect the anomaly like `One Class SVM` or `Neural Network`.

