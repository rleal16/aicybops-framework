import pandas as pd
import numpy as np
import os

# Start and end times of the attack within the collection, used to categorize the data into normal and attack traces.
START_ATTACK_TIMESTAMP = 1600643969
END_ATTACK_TIMESTAMP = 1600643993

# Files Paths
dataset_filepath = "D:/data/Security-Data/attack-traces/news-rute-wl/exploit-4/collection-1/data/sysdig.dat"
dictionary_filepath = "D:/data/Security-Data/attack-traces/news-rute-wl/exploit-4/collection-1/data/dictionary.dat"
dataset_encoded_classified_filepath = "D:/data/Security-Data/attack-traces/news-rute-wl/exploit-4/collection-1/data/encoded_dataset.dat"
dataset_bag_system_calls = "D:/data/Security-Data/attack-traces/news-rute-wl/exploit-4/collection-1/data/dataset_BoSC.dat"

# Size of the vectors
vector_size = 50

# Encodes the dataset using a dictionary, transforming system calls into numerical values, thereby handling categorical data.
def one_hot_encoding(dataset_filepath, dictionary_filepath):
    print("Start of the Encoding")
    # By using only the two columns, we obtain the timestamp and the system call name.
    # This removes unnecessary data and eliminates system call parameters, which are difficult to encode and are typically ignored.
    dataset = pd.read_csv(dataset_filepath, sep="\s+", header=None, usecols=[0,2], names=["timestamp", "system_call"])

    # Create two new columns to the dataset: encoding and label
    # for label: 0 - normal data | 1 - attack data
    dataset["encoding"] = 0
    dataset["label"] = 0

    # If the dictionary already exists, it is used as a reference; otherwise, a new dictionary file is created.
    if os.path.exists(dictionary_filepath):
        dictionary = pd.read_csv(dictionary_filepath, sep="\s+")
    else:
         dictionary = pd.DataFrame(columns = ["id", "system_call"])


    # Iterates through the dataset to transform system calls using the dictionary.
    for row  in dataset.itertuples():
        idx = row.Index

        system_call = dictionary[dictionary["system_call"] == row.system_call]
        if system_call.empty:
            last_id = 1 if pd.isna(dictionary["id"].max()) else dictionary["id"].max() + 1          # Checks the last ID in the dictionary and uses 1 if the dictionary is empty.
            new_entry = pd.DataFrame([{"id": last_id, "system_call": row.system_call}])             # Creates the new row to be appended
            dictionary = pd.concat([dictionary, new_entry], ignore_index=True)                      # Append new line to the dictionary

            dataset.at[idx, "encoding"] = last_id                                                   # Add to the dataset row the enconding from the created row in the dictionary
        else:
            dataset.at[idx, "encoding"] = system_call.iloc[0].id

        # Since we are already iterating over the dataset, we also classify each entry as either a normal or an attack trace using the label column.
        classify_trace(dataset, row, idx)

    dictionary.to_csv(dictionary_filepath, sep=" ", index=False)
    dataset.to_csv(dataset_encoded_classified_filepath, sep=" ", index=False)

    print("End of the Encoding")
    return dataset, dictionary

# Verify the timestamp of the system call to determine whether it belongs to an attack trace or a normal trace.
def classify_trace(dataset, trace, idx):
    if trace.timestamp >= START_ATTACK_TIMESTAMP and trace.timestamp <= END_ATTACK_TIMESTAMP:
        dataset.at[idx, "label"] = 1
    else:
        dataset.at[idx, "label"] = 0

# From the encoded and labeled dataset, we can now transform the data into vectors using the Bag-of-System-Calls representation. Each vector has a fixed size of 50, meaning that every vector corresponds to a group of 50 consecutive rows from the dataset.
def bag_system_calls(dataset, dictionary):
    print("Started the Vector BoSC Creation")
    chunks = []
    dictionary_size = dictionary["id"].max() + 1


    # Divide the dataset into chunks of vector_size
    for i in range(0, len(dataset), vector_size):
        chunk = dataset.iloc[i:i + vector_size]
        chunks.append(chunk)

    # Transform the chunks into vectors counting the appearance of a system call for each system call in the dictionary and verify their label. If at least one row is from an attack all vector will be a attack vector.
    vectors = []
    chunk_labels = []
    for chunk in chunks:                       
        vec = chunk_to_bosc_vector(chunk, dictionary_size)  
        vectors.append(vec)
        label = int((chunk["label"] == 1).any())
        chunk_labels.append(label)


    # Transform the vectors array into a dataframe and drop the first column, since our system calls dictionary starts at 1 and not zero
    dataset_bosc = pd.DataFrame(vectors)
    dataset_bosc = dataset_bosc.iloc[:, 1:]
    dataset_bosc["label"] = chunk_labels

    dataset_bosc.to_csv(dataset_bag_system_calls, sep=" ", index=False)
    print("Ended the Vector BoSC Creation")


# Counts the number of a system call is called in the chunk and adds it to a vector
def chunk_to_bosc_vector(chunk, num_syscalls):
    calls = chunk["encoding"].to_numpy(dtype=int)
    vec = np.bincount(calls, minlength=num_syscalls)
    return vec
    

if __name__ == "__main__":
    ds, dic = one_hot_encoding(dataset_filepath, dictionary_filepath)
    bag_system_calls(ds,dic)