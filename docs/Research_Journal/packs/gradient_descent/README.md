# Data pack: gradient-descent editing, hand trials (2026-10-03)

`session_20261003_015115.json` is the Hybrid tab's auto-saved session history from the hand trials described in `docs/Research_Journal/28.md` (section 6). It holds 14 runs on GPT-2 small, layer 8 (`gpt2-small-res-jb`), one GTX 1660 Ti:

- 11 runs with Editing method = Gradient descent (record field `mode` = "Gradient descent (per-feature)"). Each has the settings used, the rank / probability / top-1 after the edit through the real hook, the tuning-pass rank (`gradient_descent.tuning_path_rank`, equal to `real_path_rank` in all 11), the KL side effect, the number of muted and boosted features, the rank at every step (`rank_progression`), and every candidate's final multiplier (`gradient_descent.multipliers`, feature id to multiplier).
- 3 runs with the fixed sweep (`mode` = "Hybrid Mute & Boost"): "The sky is" to "falling" and to "powder" (mute 0.3 / boost 0.5), and "powder" again at 0.6 / 0.8.

Not in this file: the sweep runs for "The sky is" to "black", "Frankie Lee Sims died at" to "Dallas", "Kaka professionally plays the sport" to "soccer", "My nephew is a" to "rapist" and "The sky is" to "Adi". Those sweep numbers in the journal were read from screenshots and are marked with an asterisk there.

These are single hand-picked prompts, not a controlled experiment. Do not quote a rate from them.
