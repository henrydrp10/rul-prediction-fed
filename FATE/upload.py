import os
from pipeline.backend.pipeline import PipeLine

pipeline_upload = PipeLine().set_initiator(role='guest', party_id=1001)
pipeline_upload = pipeline_upload.set_roles(guest=1001, host=[1002, 1003, 1004, 1005], arbiter=9000)

data_base = "/vol/bitbucket/hdr21/rul-prediction-fed/federated-learning/FL-data/decision-trees/fd001/scaled/"

# Experiment sample:
# 2 Workers -> 50-50, 70-30, 90-10
# 3 Workers -> 40-30-30, 50-40-10, 70-20-10
# 5 Workers -> 20-20-20-20-20, 40-30-10-10-10, 60-10-10-10-10

parties = [1001, 1002, 1003, 1004, 1005]
num_workers = [5]
combinations = {
    "2": ["90-10"],
    "3": ["33-33-33"],
    "5": ["12-12-12-12-12"]
}

for part in num_workers:
    comb_list = combinations[str(part)]
    for comb in comb_list:
        for part_idx in range(part):
            curr_path = str(part) + " workers/" + comb + "/train_partition_" + str(part_idx) + ".csv"
            print(curr_path)
            print(f"party_" + str(part_idx) + "_train_" + comb)
            pipeline_upload.add_upload_data(file=os.path.join(data_base, curr_path),
                                            table_name=f"party_" + str(part_idx) + "_train_" + comb,
                                            namespace=f"experiment",
                                            head=1, partition=part)
            curr_path = str(part) + " workers/" + comb + "/test_partition_" + str(part_idx) + ".csv"
            print(curr_path)
            print(f"party_" + str(part_idx) + "_test_" + comb)
            pipeline_upload.add_upload_data(file=os.path.join(data_base, curr_path),
                                            table_name=f"party_" + str(part_idx) + "_test_" + comb,
                                            namespace=f"experiment",
                                            head=1, partition=part)


pipeline_upload.upload(drop=1)
