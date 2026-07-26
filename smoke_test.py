import sys
sys.path.insert(0, ".")

from src.sae_utils import load_model_and_sae
from src.editing import check_combination_safe, get_target_token_id

print("Loading model on cpu...")
model, sae = load_model_and_sae(device="cpu")

prompt = "The location of Massachusetts Institute of Technology is in"
target_str = " Cambridge"
target_token_id = get_target_token_id(model, target_str)

# Same feature IDs from the earlier Cambridge run that showed the bug
mute_feature_ids = [8459, 21169, 14430, 4445, 6863]
boost_feature_ids = [3076, 19288, 8239, 24181, 13505]

result = check_combination_safe(
    model, sae, prompt,
    mute_feature_ids=mute_feature_ids, mute_strength=0.7,
    boost_feature_ids=boost_feature_ids, boost_strength=0.7,
    target_token_id=target_token_id, top_k=10
)

print("Running check_combination_safe...")
print(result)