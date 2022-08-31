from pipeline.backend.pipeline import PipeLine
from pipeline.component import Reader, DataTransform, Intersection, HomoSecureBoost, Evaluation
from pipeline.interface import Data

# parties = [1001, 1002, 1003, 1004, 1005]
# num_workers = [2, 3, 5]
# combinations = {
#     "2": ["50-50", "70-30", "90-10"],
#     "3": ["40-30-30", "50-40-10", "70-20-10"],
#     "5": ["20-20-20-20-20", "40-30-10-10-10", "60-10-10-10-10"]
# }

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

        pipeline = PipeLine().set_initiator(role='guest', party_id=1001)
        pipeline = pipeline.set_roles(guest=1001, host=parties[1:part], arbiter=9000)

        reader_0 = Reader(name="reader_0")

        for part_idx in range(part):
            role = "guest" if part_idx == 0 else "host"
            reader_0.get_party_instance(
                role = role,
                party_id = parties[part_idx]
            ).component_param(
                table={
                    "name": "party_" + str(part_idx) + "_train_" + comb,
                    "namespace": "experiment"
                }
            )

        data_transform_0 = DataTransform(name="data_transform_0")

        for part_idx in range(part):
            role = "guest" if part_idx == 0 else "host"
            data_transform_0.get_party_instance(
                role = role,
                party_id =  parties[part_idx]
            ).component_param(
                with_label=True,
                output_format="dense",
                label_type="float"
            )

        homo_secureboost_0 = HomoSecureBoost(name="homo_secureboost_0",
                                             learning_rate=0.2,
                                             num_trees=15,
                                             task_type='regression',
                                             objective_param={"objective": "lse"},
                                             tree_param={"max_depth": 8,
                                                        "min_leaf_node": 40})

        evaluation_0 = Evaluation(name="evaluation_0", eval_type="regression")
        pipeline.add_component(reader_0)
        pipeline.add_component(data_transform_0, data=Data(data=reader_0.output.data))
        pipeline.add_component(homo_secureboost_0, data=Data(train_data=data_transform_0.output.data))
        pipeline.add_component(evaluation_0, data=Data(data=homo_secureboost_0.output.data))
        pipeline.compile()

        pipeline.fit()
        pipeline.dump("pipeline_saved_fd001_" + comb + ".pkl")

        pipeline = PipeLine.load_model_from_file("pipeline_saved_fd001_" + comb + ".pkl")
        pipeline.deploy_component([pipeline.data_transform_0, pipeline.homo_secureboost_0])

        reader_1 = Reader(name="reader_1")

        for part_idx in range(part):
            role = "guest" if part_idx == 0 else "host"
            reader_1.get_party_instance(
                role = role,
                party_id = parties[part_idx]
            ).component_param(
                table={
                    "name": "party_" + str(part_idx) + "_test_" + comb,
                    "namespace": "experiment"
                }
            )

        predict_pipeline = PipeLine()
        predict_pipeline.add_component(reader_1)
        predict_pipeline.add_component(pipeline, data=Data(predict_input={pipeline.data_transform_0.input.data: reader_1.output.data}))
        predict_pipeline.add_component(evaluation_0, data=Data(data=pipeline.homo_secureboost_0.output.data))
        predict_pipeline.predict()
