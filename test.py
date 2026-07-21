import sys
import torch
import torch.nn.functional as F
from src.sae_utils import load_model_and_sae
from src.editing import get_top_competitor_features, get_top_target_features
from src.hooks import make_ablation_hook, make_joint_ablation_hook, make_signed_ablation_hook

def run_trace(model, sae, prompt: str, target_str: str, strength: float = 0.3, max_rounds: int = 5, top_n: int = 20):
    # Ensure target_str has leading space for tokenizer matching
    if not target_str.startswith(" "):
        target_str = " " + target_str

    tokens = model.to_tokens(prompt)
    target_token_id = model.to_single_token(target_str)
    hook_name = getattr(sae.cfg, "hook_name", "blocks.8.hook_resid_pre")

    features_ablated = []
    model.reset_hooks()

    print("=" * 80)
    print(f"RUNNING TRACE:")
    print(f"  PROMPT:     '{prompt}'")
    print(f"  TARGET:     '{target_str}' (Token ID: {target_token_id})")
    print(f"  STRENGTH:   {strength}")
    print(f"  MAX ROUNDS: {max_rounds}")
    print(f"  TOP N SAES: {top_n}")
    print("=" * 80 + "\n")

    # Trace round-by-round
    for r in range(0, max_rounds + 1):
        with torch.no_grad():
            logits = model(tokens)
        probs = F.softmax(logits[0, -1, :], dim=-1)
        
        top5_probs, top5_indices = torch.topk(probs, k=5)
        current_top1_id = top5_indices[0].item()
        current_top1_str = model.to_string([current_top1_id])
        target_prob = probs[target_token_id].item()
        
        print(f"--- [ROUND {r}] Current State ---")
        if r > 0:
            print(f"Active Ablated Features: {features_ablated}")
        else:
            print("Active Ablated Features: None (Clean Baseline)")
            
        print("Top 5 Predictions:")
        for rank_idx, (p, idx) in enumerate(zip(top5_probs, top5_indices), start=1):
            token_str = model.to_string([idx.item()])
            is_target_label = " <-- [CORRECT TARGET]" if idx.item() == target_token_id else ""
            print(f"  Rank {rank_idx}: '{token_str}' (ID: {idx.item():<5}) -> {p.item()*100:.2f}%{is_target_label}")
            
        print(f"Target '{target_str}' Probability: {target_prob*100:.2f}%")
        
        if current_top1_id == target_token_id:
            print(f"\nSUCCESS! Target token '{target_str}' reached Top-1 in Round {r}!")
            break
            
        if r == max_rounds:
            print(f"\nREACHED MAX ROUNDS ({max_rounds}). Stopping trace.")
            break

        # Find competitor feature to ablate for next round
        print(f"\nScanning active features driving current Top-1 wrong guess '{current_top1_str}'...")
        competitor_features = get_top_competitor_features(model, sae, prompt, current_top1_id, top_n=top_n)
        
        selected_feature = None
        selected_delta = None
        for fid, delta in competitor_features:
            if fid not in features_ablated:
                selected_feature = fid
                selected_delta = delta
                break
                
        if selected_feature is None:
            print("No more features found to ablate. Stopping trace.")
            break
            
        features_ablated.append(selected_feature)
        print(f"Selected Feature {selected_feature} for ablation (Drops '{current_top1_str}' prob by {selected_delta*100:.2f}% at strength 1.0)")
        print("-" * 80 + "\n")
        
        # Apply persistent hooks for next round
        model.reset_hooks()
        for fid in features_ablated:
            hook_fn = make_ablation_hook(fid, sae, strength=strength)
            model.add_hook(hook_name, hook_fn)

    model.reset_hooks()
    print("=" * 80)
    print("TRACE COMPLETE")
    print("=" * 80 + "\n")

def run_batch_trace(model, sae, prompt: str, target_str: str, batch_sizes: list[int], strength: float = 0.3, top_n: int = 20):
    """
    Tests compound ablation: for each batch_size in batch_sizes, selects the top
    `batch_size` competitor features (ranked by their effect on the CURRENT top-1
    wrong token, computed ONCE at the clean baseline — not re-ranked after each
    addition), ablates all of them simultaneously, and reports the resulting
    top-5 + target probability.
    """
    if not target_str.startswith(" "):
        target_str = " " + target_str
    tokens = model.to_tokens(prompt)
    target_token_id = model.to_single_token(target_str)
    hook_name = getattr(sae.cfg, "hook_name", "blocks.8.hook_resid_pre")

    model.reset_hooks()
    with torch.no_grad():
        logits = model(tokens)
    probs = F.softmax(logits[0, -1, :], dim=-1)
    current_top1_id = torch.argmax(probs).item()
    current_top1_str = model.to_string([current_top1_id])

    print(f"\nBASELINE top-1: '{current_top1_str}' | Target '{target_str}' prob: {probs[target_token_id].item()*100:.2f}%\n")

    # Rank competitor features ONCE against the baseline top-1 token
    competitor_features = get_top_competitor_features(model, sae, prompt, current_top1_id, top_n=top_n)
    ranked_ids = [fid for fid, _ in competitor_features]

    for n in batch_sizes:
        if n > len(ranked_ids):
            print(f"WARNING: requested batch size {n} exceeds {len(ranked_ids)} available active features — using all {len(ranked_ids)} instead.")

        batch = ranked_ids[:n]
        model.reset_hooks()
        joint_hook_fn = make_joint_ablation_hook(batch, sae, strength=strength)
        model.add_hook(hook_name, joint_hook_fn)

        with torch.no_grad():
            logits = model(tokens)
        probs = F.softmax(logits[0, -1, :], dim=-1)
        top5_probs, top5_indices = torch.topk(probs, k=5)
        new_top1_id = top5_indices[0].item()
        new_top1_str = model.to_string([new_top1_id])
        target_prob = probs[target_token_id].item()

        print(f"--- BATCH SIZE {n} (features: {batch}) ---")
        print(f"  New top-1: '{new_top1_str}' | Target '{target_str}' prob: {target_prob*100:.2f}%")
        for rank_idx, (p, idx) in enumerate(zip(top5_probs, top5_indices), start=1):
            tok = model.to_string([idx.item()])
            flag = " <-- TARGET" if idx.item() == target_token_id else ""
            print(f"    Rank {rank_idx}: '{tok}' -> {p.item()*100:.2f}%{flag}")
        print()

    model.reset_hooks()

def run_boost_trace(
    model, sae, prompt: str, target_str: str, boost_strengths: list[float], batch_sizes: list[int], top_n: int = 30
):
    """
    Tests target feature amplification (boosting): for each combination of boost strength and batch size,
    finds the top target-supporting features, turns them UP using make_signed_ablation_hook,
    and reports the resulting top-5 + target probability.
    """
    if not target_str.startswith(" "):
        target_str = " " + target_str
    tokens = model.to_tokens(prompt)
    target_token_id = model.to_single_token(target_str)
    hook_name = getattr(sae.cfg, "hook_name", "blocks.8.hook_resid_pre")

    model.reset_hooks()
    with torch.no_grad():
        logits = model(tokens)
    probs = F.softmax(logits[0, -1, :], dim=-1)
    current_top1_id = torch.argmax(probs).item()
    current_top1_str = model.to_string([current_top1_id])

    print(f"\nBASELINE top-1: '{current_top1_str}' | Target '{target_str}' prob: {probs[target_token_id].item()*100:.2f}%\n")

    # Rank target features ONCE against clean baseline
    target_features = get_top_target_features(model, sae, prompt, target_token_id, top_n=top_n)
    ranked_ids = [fid for fid, _ in target_features]

    for strength in boost_strengths:
        for n in batch_sizes:
            if n > len(ranked_ids):
                print(f"WARNING: requested batch size {n} exceeds {len(ranked_ids)} available active features — using all {len(ranked_ids)} instead.")

            batch = ranked_ids[:n]
            model.reset_hooks()
            signed_hook_fn = make_signed_ablation_hook(batch, sae, strength=strength)
            model.add_hook(hook_name, signed_hook_fn)

            with torch.no_grad():
                logits = model(tokens)
            probs = F.softmax(logits[0, -1, :], dim=-1)
            top5_probs, top5_indices = torch.topk(probs, k=5)
            new_top1_id = top5_indices[0].item()
            new_top1_str = model.to_string([new_top1_id])
            target_prob = probs[target_token_id].item()

            print(f"--- BOOST STRENGTH +{strength} | BATCH SIZE {n} (features: {batch}) ---")
            print(f"  New top-1: '{new_top1_str}' | Target '{target_str}' prob: {target_prob*100:.2f}%")
            for rank_idx, (p, idx) in enumerate(zip(top5_probs, top5_indices), start=1):
                tok = model.to_string([idx.item()])
                flag = " <-- TARGET" if idx.item() == target_token_id else ""
                print(f"    Rank {rank_idx}: '{tok}' -> {p.item()*100:.2f}%{flag}")
            print()

    model.reset_hooks()

def main():
    print("=" * 80)
    print("FEATURE SCALPEL — PERSISTENT INTERACTIVE REPL BENCH")
    print("================================================================================")
    
    layer_in = input("Which layer? [8]: ").strip()
    if layer_in.lower() in ["q", "quit", "exit"]:
        return
    
    layer = 8
    if layer_in:
        try:
            layer = int(layer_in)
        except ValueError:
            print("Invalid layer number. Defaulting to Layer 8.")
            layer = 8

    print(f"Loading GPT-2-small and SAE (Layer {layer}) into memory ONCE...\n")
    model, sae = load_model_and_sae(device="cpu", layer=layer)
    print(f"\nModel & SAE (Layer {layer}) loaded successfully! Memory ready. Tests will now execute INSTANTLY.")
    print("================================================================================")

    # Defaults
    last_prompt = "Seiyu Group's headquarters are in"
    last_target = "Tokyo"
    last_strength = 0.3
    last_max_rounds = 5
    last_top_n = 30

    while True:
        print("\n" + "-" * 80)
        print("CONFIGURE TEST CASE (Press Enter to keep default/previous | Type 'q' to quit)")
        print("-" * 80)
        
        try:
            p_in = input(f"Prompt [{last_prompt}]: ").strip()
            if p_in.lower() in ["q", "quit", "exit"]:
                break
            if p_in: last_prompt = p_in

            t_in = input(f"Target [{last_target}]: ").strip()
            if t_in.lower() in ["q", "quit", "exit"]:
                break
            if t_in: last_target = t_in

            s_in = input(f"Ablation/Boost Strength [{last_strength}]: ").strip()
            if s_in.lower() in ["q", "quit", "exit"]:
                break
            if s_in: 
                try: last_strength = float(s_in)
                except ValueError: print("Invalid strength float. Keeping previous value.")

            m_in = input(f"Max Rounds [{last_max_rounds}]: ").strip()
            if m_in.lower() in ["q", "quit", "exit"]:
                break
            if m_in: 
                try: last_max_rounds = int(m_in)
                except ValueError: print("Invalid max rounds int. Keeping previous value.")

            if "dublin" in last_target.lower() and last_top_n < 30:
                last_top_n = 30

            n_in = input(f"Top N SAE Features Candidate Pool [{last_top_n}]: ").strip()
            if n_in.lower() in ["q", "quit", "exit"]:
                break
            if n_in: 
                try: last_top_n = int(n_in)
                except ValueError: print("Invalid top_n int. Keeping previous value.")

            print("\nSelect Mode:")
            print("  1: Single-trace iterative ablation (mute wrong guess)")
            print("  2: Compound batch test (mute wrong guess)")
            print("  3: Target feature boost (amplify correct answer)")
            mode_in = input("Mode [1]: ").strip().lower()
            if mode_in in ["q", "quit", "exit"]:
                break

            print()
            if mode_in == "2":
                bs_in = input("Batch sizes comma-separated [1,3,5,10,15,20]: ").strip()
                if bs_in.lower() in ["q", "quit", "exit"]:
                    break
                if bs_in:
                    try:
                        batch_sizes = [int(x.strip()) for x in bs_in.split(",") if x.strip()]
                    except ValueError:
                        print("Invalid batch sizes format. Using default [1, 3, 5, 10, 15, 20].")
                        batch_sizes = [1, 3, 5, 10, 15, 20]
                else:
                    batch_sizes = [1, 3, 5, 10, 15, 20]
                
                run_batch_trace(model, sae, prompt=last_prompt, target_str=last_target, batch_sizes=batch_sizes, strength=last_strength, top_n=last_top_n)
            elif mode_in == "3":
                str_in = input("Boost strengths comma-separated [0.3, 0.5, 0.7, 1.0]: ").strip()
                if str_in.lower() in ["q", "quit", "exit"]:
                    break
                if str_in:
                    try:
                        boost_strengths = [float(x.strip()) for x in str_in.split(",") if x.strip()]
                    except ValueError:
                        print("Invalid boost strengths. Using default [0.3, 0.5, 0.7, 1.0].")
                        boost_strengths = [0.3, 0.5, 0.7, 1.0]
                else:
                    boost_strengths = [0.3, 0.5, 0.7, 1.0]

                bs_in = input("Batch sizes comma-separated [1,3,5]: ").strip()
                if bs_in.lower() in ["q", "quit", "exit"]:
                    break
                if bs_in:
                    try:
                        batch_sizes = [int(x.strip()) for x in bs_in.split(",") if x.strip()]
                    except ValueError:
                        print("Invalid batch sizes. Using default [1, 3, 5].")
                        batch_sizes = [1, 3, 5]
                else:
                    batch_sizes = [1, 3, 5]

                run_boost_trace(model, sae, prompt=last_prompt, target_str=last_target, boost_strengths=boost_strengths, batch_sizes=batch_sizes, top_n=last_top_n)
            else:
                run_trace(model, sae, prompt=last_prompt, target_str=last_target, strength=last_strength, max_rounds=last_max_rounds, top_n=last_top_n)

        except (KeyboardInterrupt, EOFError):
            break

    print("\nExiting Feature Scalpel Interactive Bench. Goodbye!")

if __name__ == "__main__":
    main()
