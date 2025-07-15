import os
import json

# Directory to save PDFs
def traverse_folder(folder_path):
    """
    Traverse through the folder and retrieve file names.

    Args:
        folder_path (str): Path to the folder to traverse.

    Returns:
        list: A list of file paths.
    """
    file_list = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list


total_recall_300 = 0
total_nDCG_300 = 0
total_recall_200 = 0
total_nDCG_200 = 0
total_recall_100 = 0
total_nDCG_100 = 0
total_test_sets = 0

ours = ""
file_list = traverse_folder(ours)
for file in file_list:
    with open(file, "r") as json_file:
        data = json.load(json_file)
    try:
        current_results = data["Current Result"]
    except:
        continue
    total_test_sets = total_test_sets + 1
    
    recall300 = current_results["Recall@300"]
    total_recall_300 = total_recall_300 + recall300
    ndcg300 = current_results["nDCG@300"]
    total_nDCG_300 = total_nDCG_300 + ndcg300
    
    recall200 = current_results["Recall@200"]
    total_recall_200 = total_recall_200 + recall200
    ndcg200 = current_results["nDCG@200"]
    total_nDCG_200 = total_nDCG_200 + ndcg200
    
    recall100 = current_results["Recall@100"]
    total_recall_100 = total_recall_100 + recall100
    ndcg100 = current_results["nDCG@100"]
    total_nDCG_100 = total_nDCG_100 + ndcg100
    
print(f"Total number of samples taken into account are {total_test_sets}")
if total_test_sets != 400:
    raise ValueError("Number of finished files is not 400")

average_recall_300 = total_recall_300 / total_test_sets
average_nDCG_300 = total_nDCG_300 / total_test_sets
print(f"Average Recall@300 is {average_recall_300}")
print(f"Average nDCG@300 is {average_nDCG_300}")

average_recall_200 = total_recall_200 / total_test_sets
average_nDCG_200 = total_nDCG_200 / total_test_sets
print(f"Average Recall@200 is {average_recall_200}")
print(f"Average nDCG@200 is {average_nDCG_200}")

average_recall_100 = total_recall_100 / total_test_sets
average_nDCG_100 = total_nDCG_100 / total_test_sets
print(f"Average Recall@100 is {average_recall_100}")
print(f"Average nDCG@100 is {average_nDCG_100}")