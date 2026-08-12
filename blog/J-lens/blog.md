---
title: "Investigating the J-Lens in Mixture-of-Experts Models"
description: A journey through the theory and practice of the J-Lens in Mixture-of-Experts Models
date: 2026-08-10
permalink: /blog/j-lens/
image: /blog/j-lens/j-lens.png
---

# Investigating the J-Lens in Mixture-of-Experts Models

## 1. Introduction

In July, Antropic provided evidence that claude has "access conciousness". WE are doomed. 

Anyway, in this work they use a new mechanistic interpability technique called the Jacobian lens to identify a "global workspace", a space where the model "maintains a priviledge set of internal representations, available for report, modulation, and flexible internal reasoning, atop a much larger volume of automatic processing". The full paper is here: [Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html) 

The point of the J-lens is to identify what is verbalizable at a given point in the models computation (at some Layer l, and token position t). Extremely collouquially, we could interpret this as reading the models mind at a given point in time. Please Anthropic, don't send a death squad after me, I'm just a lowly blogger.

"Ellington, I'm to lazy to read the paper, because I'm a giga chud. What is exactly is the J-lens and how is it constructed?"

I'm gald you asked. The J-lens maps the hidden state $h_{l, t}$ to scores $s_{l, t}$ over the vocabulary, characterizing the activation's average first-order causal effect on the model's output logits at the current and subsequent token positions, averaged across contexts. It answers the question: "what is the average causal effect of the hidden state at position $t$ on the future outputs?"

When constructing the lense, remember we want the lens to be general in the sense we can apply it to a variety of contexts, so we  average over a diverse corpus of contexts. Within each context, we compute the affect the hidden state at position $t$ has on the future outputs at position $t'$. This affect is captured by the Jacobian matrix $\frac{\partial h_{\text{final}, t'}}{\partial h_{l, t}}$. We average this over all contexts to get the final J-lens.

$$J_{l} = \mathbb{E}_{t, t' \geq t, \text{corpus}}\left[\frac{\partial h_{\text{final}, t'}}{\partial h_{l, t}}\right]$$

<figure class="post-figure" markdown="0">
<img src="computing_J-lens.png" alt="J-lens" width="800">
<figcaption>Figure from Anthropic illustrating computing the J-lens</figcaption>
</figure>


Anthropic did this great work, but why do we care? Well, averaging a Jacobian across positions and prompts is a perfectly sensible thing to do if layer 1 is one function. You're sampling one map at many points and taking the mean. Fine. Whatever. But in Mixture-of-Experts models there may be up to 384 functions per layer. And more problematic still, each token activates some subset of those functions, whose outputs are then combined into the layer's output. Even if only 2 were active at a time, that's more than 70,000 possible combinations per token. It is not at all clear that averaging over this, on top of averaging over tokens and prompts, is valid.

(As an aside, I see the fact this problem was not addressed in the paper as good evidence the models used Opus and Sonnet are dense rather than MoE. I think this is the consensus but they are not open so who knows)

A natural set of questions is: 

- ***Is a naively averaged jacobian lense valid?***

- ***How much does the averaged J-len deviate from a MoE aware lense?***

- ***When we have different functions per-token, per-layer, where does this global workspace live?***

- ***The workspace may be regime-local rather than global. is this true?***

- ***Is routing itself workspace-sensitive?*** 

This blog will pertain to the first 2. Future blogs will deal the last 3. 

## 2. Building a MoE Aware Jacobian

For my experiments, I used DeepSeek-V2-Lite-Chat. This was the largest model I could use with my compute constraints (1 h100).  DeepSeek-V2-Lite-Chat has both an always-on shared-expert pathway and 64 routed experts. 

<figure class="post-figure" markdown="0">
<img src="MoE.png" alt="J-lens" width="800">
<figcaption>MoE layer</figcaption>
</figure>

My first idea was to build jacobians conditioned on the top-k routing pattern of layer l. This is a really stupid idea. For a LOT of reasons. Lets assume we have top-6 is a good enough approximation, we now have $\binom{64}{6} = 7 * 10^7$ jacobians to build, just not posible. On top of that, consider conditioning on a specific routing pattern: This is equivelant to conditioning on the input tokens themselves. Tokens route to combination C precisely because their $h_{l, t} occupies a particular region of activation space. So two pattern-conditional Jacobians can differ even if the experts were byte-identical, just because they see different tokens. 

Once I locked in, it became apparent we should build the jacobians per-expert rather than per top-k pattern. The combinatorial difficulty of this first problem alerted me to another: how do we deal with the routing of future layers. Routing does not just happen at the layer the jacobian is built for. It happens at each layer. 

I don't solve that here. So its maybe more accurate to call this a partially MoE aware jacobian. What I do instead is factor it into a decomposition where I can name and study routing aware and the non-routing aware portion seperately:

$$J_{\ell,t} = D_t M_{\ell,t}, \qquad M_{\ell,t} = I + \sum_{e \in S_t} g_{t,e} J_{f_e}(h_{\ell,t})$$

$M_{\ell,t}$ is the **inner map**, layer $\ell$'s input to its output, with $I$ the residual skip, $S_t$ the experts this token selected, and $g_{t,e}$ their gate weights. Being MoE aware is built into this map, it considers the per-expert jacobians at this layer. It's exactly linear in the per-expert Jacobians $J_{f_e}$, which is what makes the whole thing tractable. 

$D_t$ is the **downstream map**: everything after layer $\ell$'s output, including all downstream routing and the position-mixing that attention does across $t' \geq t$. It's computed by autodiff on the real forward pass, so downstream routing isn't approximated away, it's baked in at whatever routing actually fired.

 $J_{\ell,t}$ is the **full path**, it's what actually reaches the output, and it's analagous to the object the original paper's claims are about. The inner map and the per-expert pieces are there to *decompose and attribute* it. I never treat $M_{\ell,t}$ on its own as the workspace, because a difference that's large locally can be damped to nothing by $D_t$, and a small one can be amplified.

Two things fall out of this for free. Firstly, expanding the product splits the Jacobian additively into a residual term, a shared-expert term, and a routed term:

$$J_{\ell,t} = \underbrace{D_t}_{\text{residual}} + \underbrace{\sum_{e \in \text{shared}} g_{t,e} D_t J_{f_e}}_{\text{shared}} + \underbrace{\sum_{e \in S_t^{\text{routed}}} g_{t,e} D_t J_{f_e}}_{\text{routed}}$$

Notice that because of this decomposition we can probe/measure where the workspace lives (at least the broadcast requirement) without having to zero out experts, and work with a broken model. 

Second, I can average each $J_{f_e}$ two ways: $\bar J^{\text{on}}_e$, over the tokens that actually routed to $e$, and $\bar J^{\text{common}}_e$, over a common probe set every expert sees. The gap between those is the input-niche confound I complained about two paragraphs ago, We can now measure it. 


<figure class="post-figure" markdown="0">
<img src="expert_jacobian.png" alt="J-lens" width="800">
<figcaption>Computing the MoE Aware J-lens and Expert Jacobians</figcaption>
</figure>

## 3. Beating away the many Confounds 

OK, so we've done all this set up, where is the climax? (ayo) Is the native jacobian valid or not?  Before we get there, I am going to try to kill off all the confounding variables that may take away from the headline. 

### What does it mean for the jacobian lense to be valid? 

It means a native Jacobian lense with no knowledge of the experts aligns with a lense that does have knowledge. Now, you are probably thinking "Ellington, this is stupid. You're stupid.  Autodiff differentiates whatever forward pass actually ran including the router and experts, so native J already has the routing baked in. dummy." and you are right but there is a subtlty here. Remember that the Jacobian is computed by taking a mean over the tokens. but, each token is processed by a set of discrete and structually seperate sub-networks. It is entirely possible that a mean over these tokens does not summarize the population accurately and instead is a centroid that lives in no man's land. Imagine the mean of a wide bimodal distribution its possible the mean lives in a place no member of the distribution lives. 

<figure class="post-figure" markdown="0">
<img src="ellington.png" alt="J-lens" width="125">
<figcaption>My readers opinion of me</figcaption>
</figure>


### First, is our decomposition valid? What is the role of $D_t$? (Confound 1)
Prior to directly measuring the MoE aware and non-MoE aware jacobians against each other, We need to do a few things. Firstly, Is this decomposition I introducd in the above section even valid? I would certainly be supicious. We only are expert aware at one layer and the downstream is completely blind. In order to test, this I compare the variation in the Jacobians before and after the downstream map is applied. The concern, stated precisely, is that $D_t$ might be routing-dependent in disguise: if the rest of the network responds differently to the output of one expert mix than another, then differences between local maps would be reshaped on their way to the output. And then we have a major confounding factor, that all of layer l's routing mixture strucutre doesn't actually live exclusively in $M_t$


<figure class="post-figure" markdown="0">
<img src="screwed.png" alt="J-lens" width="250">
<figcaption>what would happen if D_t was routing dependent</figcaption>
</figure>

In order to test this, I compute the dispersion (measured by pairwise CKA distance) of the inner maps Mₜ and of the corresponding full-path Jacobians Jₜ = Dₜ·Mₜ, and take their ratio: the full-vs-inner dispersion ratio, a downstream amplification factor. If Dₜ is neutral transport, applying it changes nothing about the relative geometry and the ratio sits at 1; a ratio above 1 means the downstream path compounds routing differences, below 1 means it equalizes them.

<figure class="post-figure" markdown="0">
<img src="full_inner_dispertion.png" alt="J-lens" width="600">
<figcaption>Dispertion</figcaption>
</figure>

 If an expert-blind $D_t$ were distorting the picture, the spread of the full-path Jacobians $J_{\ell,t} = D_t M_{\ell,t}$ would systematically depart from the spread of the inner maps alone. The downstream path would either exaggerate routing differences on their way to the output or wash them out, and in either case the decomposition would be attributing structure to the wrong factor. At the expert level it does neither: the ratio sits at ~1.04 at layer 1 and decays to ~1.00 by layer 26. So, differences between the per-expert averaged maps J̄ₑ arrive at the output essentially unchanged. $D_t$ acts as expert-neutral transport. (LETS GO!) All of the routing-dependence lives in Mₜ, where we measure it. The same plot, read at token granularity, is also a great vote of confidence for us. The token-level ratio is above 1 at every probed layer and grows with depth (~1.06 → ~1.24), so per-token differences in the local map are not averaged away downstream: they are mildly amplified. Combined with between-expert dispersion sitting an order of magnitude above the split-half noise floor (seen in figure TODO), this means the naive global $J_l$ is blending linear maps that genuinely differ, and the blend gets less representative of any individual token's path the deeper the layer. The naive average may survive as a coarse summary, but as a per-token object it is shaky. 

 ### Are our inputs Confounding us? (Confound 2)

 If you remember earlier I mentioned a potential confunding variable while computing Jacobians unique to MoE models, which is each expert sees different sets of tokens. So even if two expert are completely identical, If we were to compute two jacobians they would be different just as a result of the differing input distributions. 

 In order to measure this confound, I compute 2 Jacobians per expert, $J_{e, on}$ and $J_{e, common}$. $J_{e, on}$ averages the expert's Jacobian over the tokens that got routed to it, and $J_{e, common}$ averages the same expert's Jacobian over a fixed, shared set of inputs that's the same for every expert, whether or not those inputs would normally be sent to it.

 The difference between them separates two things that are otherwise tangled together: what the expert intrinsically computes, versus the particular slice of inputs the router happens to feed it. $J_{e, common}$ isolates the intrinsic behavior (everyone measured on the same inputs), and the gap $ \lVert J_{e, on}$ − $J_{e, common}\rVert$ tells you how much of an expert's behavior is an artifact of its input.

 (I think it is important to note that while I call $J_{e, on}$, $J_{e, common}$ jacobians, they are not a J-lens. Neither $J_{e, on}$, $J_{e, common}$ take layer input to model output, rather they take layer input to layer output through a specific expert)

<figure class="post-figure" markdown="0">
<img src="input_gap.png" alt="J-lens" width="600">
<figcaption>Input Gap</figcaption>
</figure>

The important take away from this graph is that the gap is nontrivially large at each measured layer. Input distribution moves each expert so if we want to compare experts differences we should use a common set of tokens. And, we'll see in a future figure that the CKA gap is smaller than the difference between experts. Meaning the input gap is big enough that wecannot conflate the two averagings, and small enough that expert diversity survives controlling for it, which is the combination that makes the per-expert decomposition we did both necessary and sufficient. Yippeee. 

### The ≈ I've been hiding from you (Confound 3) FAH

You caught me. When I wrote $M_{\ell,t} = I + \sum_{e \in S_t} g_{t,e} J_{f_e}(h_{\ell,t})$ back in section 2, I treated the routing as frozen. But the gate weights $g_{t,e}$ are not constants, they're functions of the same hidden state I'm differentiating with respect to. A proper application of the product rule gives a second term, $\sum_e f_e(h)\, \frac{\partial g_e}{\partial h}$: nudging $h_{\ell,t}$ doesn't just change what each expert computes, it changes *how much the router listens to each expert*. My decomposition drops that term entirely. So the honest equation has an $\approx$ in it, and every result in this post is standing on that $\approx$. 

"Why did you do this?": Notice that if we differentiate the sublayer honestly we have a term $f_e(h)\nabla g_e$. This is a very problematic term $\nabla g_e$ envolves every expert implicitly because the router scores experts competitively, so we run into the combinatorial sampling problem we ran into with out top-k routing pattern. AAARGHGHGH!

Yeah Yeah this sucks. The good news is I never have to hand-wave about how big the dropped term is, because it is directly measurable. Autodiff gives me the true full-path Jacobian $J_{\ell,t}$, and I can separately reconstruct $$D_t M^{\text{recon}}_{\ell,t}$$ from the per-expert pieces and the logged gates, which contains everything *except* the gate-derivative term. Their disagreement, $$\lVert J_{\ell,t} - D_t M^{\text{recon}}_{\ell,t}\rVert / \lVert J_{\ell,t}\rVert$$, *is* the dropped term.

<figure class="post-figure" markdown="0">
<img src="gate_residual.png" alt="gate-derivative residual" width="600">
<figcaption>Gate-derivative residual (median relative reconstruction error)</figcaption>
</figure>

The bottom curve reconstructs each token's map from its *own* per-token expert Jacobians $J_{f_e}(h_{\ell,t})$. The residual starts around ~0.45 at layer 1 and falls to ~0.24 by layer 26: the routing-frozen approximation is rough early, the router's sensitivity to the hidden state is a real part of the layer-1 Jacobian, but it captures the large majority of the map at depth, where routing has apparently settled down.

The top curve is a little scary. Reconstruct the same tokens using the averaged $\bar J^{\text{on}}_e$ instead of the per-token expert Jacobians and the residual balloons to ~0.91 at layer 1, still ~0.69 at layer 26. Read that carefully, because it's a trap I want you to avoid: the per-expert averages are *population* objects. They're the right tool for asking what expert 41 does on average, how different the experts are from each other, where the shared pathway sits. BUT BUT BUT, They are not per-token predictors,  you cannot look up a token's six experts, grab six averaged matrices off the shelf, blend them with the gates, and expect to recover *that token's* map. The expert nonlinearities move too much with their inputs for the shelf copy to stand in for the local one.

"Wait, Ellington, if reconstructing from averages is ~90% wrong, isn't your whole validity analysis built on sand?" No, because nothing in it routes through that reconstruction. Every result here compares like with like: the token-deviation and token-level dispersion results use per-token Jacobians measured directly by autodiff on each token's real forward pass, no $\bar J_e$ anywhere in that pipeline,while the between-expert dispersion, the niche gap, and the expert-level ratio are population claims about population objects, which is exactly what averages are for. The ~0.9 residual kills a shortcut I never took: blending shelf-averaged expert matrices to fake a token's map. The reason I address it is because the previous sections read as if I use this shortcut. 

If anything, the top curve is evidence *for* the thesis. My worry was that a mean over a population can be a centroid where no member lives. The $\bar J^{\text{on}}_e$ curve is that same phenomenon one level down: even within a single expert, the average Jacobian is a poor stand-in for any particular token's copy. The bimodal-distribution picture from earlier applies here. Averages are population summaries all the way down, and "fine as a summary, shaky as a per-token object" turns out to be true of experts for exactly the reason it's true of the layer.

### 















