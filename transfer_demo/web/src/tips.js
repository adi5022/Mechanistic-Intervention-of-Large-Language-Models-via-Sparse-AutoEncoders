// All the pop-up explanations in one place, so the wording can be edited without touching the page code.
// Written for someone seeing the page for the first time: what the thing is, and what it means for this experiment.
// Entries that need a number from the current run are functions; the rest are plain text.

export const TIPS = {
  // ---- the model panels ----
  small: (f) => ({
    title: "GPT-2 Small",
    text: `The smaller model (${f.small.n_layers} layers, about ${Math.round(f.small.params_millions)} million parameters). It is the one that has an SAE, so this is where the edit is designed.`,
  }),
  medium: (f) => ({
    title: "GPT-2 Medium",
    text: `The bigger model (${f.medium.n_layers} layers, about ${Math.round(f.medium.params_millions)} million parameters). It has no SAE here. It never designs an edit; it only receives one that has been carried across from Small.`,
  }),
  rank: {
    title: "Target rank",
    text: "Where the target word stands in the model's list of next-word guesses, from most to least likely. Rank 1 means it is the model's first choice, which is the goal of the edit. \"2 → 1\" means rank 2 before the edit and rank 1 after.",
  },
  probability: {
    title: "Probability",
    text: "How likely the model thinks the target word is as the next word, out of all the words it knows. Rank 1 can still have a small probability if every other word is unlikely too.",
  },

  // ---- layers and the two special places ----
  layer: {
    title: "A layer",
    text: "One layer (transformer block) of the model. Here they are numbered from 1 at the top. The study's \"layer 8\" is the model's state after 8 layers have run.",
  },
  intervention: {
    title: "Intervention: where the edit is made",
    text: "At this layer of Small the SAE splits the model's state into features. Gradient descent then turns 200 dials, one for each feature that fires on your sentence, up or down until the target word becomes the model's first choice. (With the Direct method the dials are tuned by watching Medium's answer instead.)",
  },
  injection: {
    title: "Injection: where the edit lands",
    text: "The translated change is added to Medium's state at this layer, at every word of the sentence except the very first, scaled by the dose. Nothing in Medium is retrained or permanently changed; the change exists only for this run.",
  },
  saeFeatures: (f) => ({
    title: "SAE features",
    text: `An SAE (sparse autoencoder) is a small network that splits a model's internal state into many readable "features", each of which tends to switch on for one kind of thing. Small's SAE has ${f.small.sae_features.toLocaleString("en-US")} of them at this layer. Only the 200 that fire most strongly on your sentence get a dial.`,
  }),
  io: {
    title: "Sentence in, prediction out",
    text: "The sentence enters at the first layer, each layer refines the model's internal state, and after the last layer the model produces its guess for the next word.",
  },

  // ---- the bridge ----
  translator: (mapLabel, mapHelp, dose) => ({
    title: `Translator: ${mapLabel}`,
    text: `The two models do not describe things in the same internal coordinates, so the change made in Small has to be converted before Medium can use it. ${mapHelp} The dose (×${dose.toFixed(1)}) multiplies the converted change before it is added.`,
  }),
  state: (pct) => ({
    title: "State change",
    text: `How big the injected edit is compared with what Medium already holds. At the last word of your sentence, the change added to Medium is ${pct}% as large as Medium's own internal state at that point. A small number means a push, not a rewrite.`,
  }),

  // ---- the controls ----
  dose: {
    title: "Dose",
    text: "A multiplier on the translated change before it is added to Medium. Higher is a stronger push. 2.0 was chosen on development records for the 8 to 16 pair; other pairs were not tuned.",
  },
  layers: {
    title: "Translation layers",
    text: "The edit is made at one layer of Small and added at one layer of Medium. 8 to 16 was the main pair of the study; the others were fitted but not tuned.",
  },
  mechanism: {
    title: "Edit mechanism",
    text: "How Small's edit is made. Plain gradient descent only turns the 200 dials. The additive version can also add some of the features that are silent on your sentence, which is strong enough to push almost any word to the top, so treat it as a control.",
  },
  map: {
    title: "Translation map",
    text: "The method that carries the change from Small to Medium. Ridge, neural and trained maps carry a change that was made by looking at Small only. Direct is different: it tunes the dials by looking at Medium's own answer.",
  },
  sentence: {
    title: "Sentence and target",
    text: "The prompt both models read, and the word we want to become the first guess for what comes next. The target is usually a wrong answer, so the edit has to push against what the model believes.",
  },

  // ---- the dials ----
  dial: {
    title: "A dial",
    text: "One multiplier on one SAE feature. 1.0 means leave the feature alone, above 1 turns it up and below 1 turns it down. Gradient descent starts every dial at 1 and moves them until the target word wins.",
  },
  turnedUp: { title: "Turned up", text: "Dials with a multiplier above 1: the edit makes these features fire more strongly." },
  turnedDown: { title: "Turned down", text: "Dials with a multiplier below 1: the edit makes these features fire less strongly." },
  movers: { title: "Biggest movers", text: "The dials that moved furthest from 1. Each link opens the feature's page on Neuronpedia (needs internet), which shows what kind of text switches it on." },
};
