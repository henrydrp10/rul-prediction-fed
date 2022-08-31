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
num_workers = [3]
combinations = {
    "2": ["90-10"],
    "3": ["33-33-33"],
    "5": ["20-20-20-20-20"]
}

for part in num_workers:
    comb_list = combinations[str(part)]
    for comb in comb_list:

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

        evaluation_0 = Evaluation(name="evaluation_0", eval_type="regression")
        predict_pipeline = PipeLine()
        predict_pipeline.add_component(reader_1)
        predict_pipeline.add_component(pipeline, data=Data(predict_input={pipeline.data_transform_0.input.data: reader_1.output.data}))
        predict_pipeline.add_component(evaluation_0, data=Data(data=pipeline.homo_secureboost_0.output.data))
        predict_pipeline.predict()
